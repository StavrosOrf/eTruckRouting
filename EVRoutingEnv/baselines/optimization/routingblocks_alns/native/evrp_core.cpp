// Exact route pricing for the nominal joint electric-truck routing problem.
//
// Given a customer sequence, `price_route` returns the cheapest energy-feasible
// insertion of charger stops and discrete charge levels under the environment's
// model: per-leg energy margin, discrete target-SoC levels strictly above the
// arrival SoC, each station's own cumulative charge-time table, customer time
// windows, payload (checked by the caller), and the horizon.
//
// Label = (cost, time, soc), Pareto-ordered: lower cost, earlier time, higher soc.
//   cost = travel + w * (charging + service)   (w = 0: travel objective, 1: operating)
//   time = clock, including charging, service and waiting for a window to open.
//
// Between two consecutive stops of the customer sequence the truck may detour
// through chargers. After a charge the SoC is one of the discrete levels, so
// everything downstream of a charging stop is a function of (station, level) and
// the target only; `fronts_` precomputes those Pareto fronts once per instance.
// This is the role frvcpy plays for a continuous-charging model.
//
// Not covered by the route DP (it only costs optimality, never feasibility): a
// detour that passes through two or more chargers before the first charge.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kEps = 1e-9;

struct Lab {
    double cost, time, soc;
};

inline bool dominates(const Lab& a, const Lab& b) {
    return a.cost <= b.cost + 1e-12 && a.time <= b.time + 1e-12 && a.soc >= b.soc - 1e-12;
}

// A front point of a precomputed (station, level) -> target table.
struct FrontPt {
    double cost, time, soc;
    int path;  // index into Core::path_off_, -1 = direct
};

struct Stop {
    int node;
    int level;  // -2 customer/depot, -1 station passed through, >= 0 charged to levels_[level]
};

class Core {
public:
    Core(py::array_t<double> energy, py::array_t<double> hours, int n_customers,
         std::vector<double> demand, std::vector<double> service, std::vector<double> tw_open,
         std::vector<double> tw_close, double battery, double start_soc_kwh, double margin,
         double horizon, double payload, std::vector<double> levels,
         std::vector<std::vector<double>> curves, std::vector<int> curve_of_station,
         double curve_step, double service_cost_weight, int max_labels)
        : nc_(n_customers), B_(battery), x0_(start_soc_kwh), margin_(margin), horizon_(horizon),
          payload_(payload), levels_(std::move(levels)), curves_(std::move(curves)),
          curve_of_(std::move(curve_of_station)), step_(curve_step), w_(service_cost_weight),
          max_labels_(max_labels), demand_(std::move(demand)), service_(std::move(service)),
          open_(std::move(tw_open)), close_(std::move(tw_close)) {
        auto e = energy.unchecked<2>();
        auto h = hours.unchecked<2>();
        N_ = static_cast<int>(e.shape(0));
        if (e.shape(1) != N_ || h.shape(0) != N_ || h.shape(1) != N_) {
            throw std::invalid_argument("energy/hours must be square and equal");
        }
        ns_ = N_ - nc_ - 1;
        if (ns_ < 0) throw std::invalid_argument("n_customers too large");
        if (static_cast<int>(curve_of_.size()) != ns_) throw std::invalid_argument("curve_of_station size");
        E_.assign(static_cast<size_t>(N_) * N_, kInf);
        H_.assign(static_cast<size_t>(N_) * N_, kInf);
        for (int a = 0; a < N_; ++a) {
            for (int b = 0; b < N_; ++b) {
                double ev = e(a, b), hv = h(a, b);
                if (a != b && std::isfinite(ev) && std::isfinite(hv) && ev >= 0.0) {
                    E_[a * N_ + b] = ev;
                    H_[a * N_ + b] = hv;
                }
            }
        }
        std::sort(levels_.begin(), levels_.end());
        build_fronts();
    }

    int n_nodes() const { return N_; }
    int n_stations() const { return ns_; }
    int front_points() const {
        int total = 0;
        for (const auto& f : fronts_) total += static_cast<int>(f.size());
        return total;
    }
    int label_cap_hits() const { return cap_hits_; }

    // ---- charge-time helpers -------------------------------------------------
    double cum(int station_index, double soc_fraction) const {
        const auto& table = curves_[curve_of_[station_index]];
        double x = std::min(std::max(soc_fraction, 0.0), 1.0) / step_;
        int k = std::min(static_cast<int>(std::floor(x)), static_cast<int>(table.size()) - 2);
        return table[k] + (x - k) * (table[k + 1] - table[k]);
    }
    double charge_hours(int station_index, double from_kwh, double to_level_fraction) const {
        return cum(station_index, to_level_fraction) - cum(station_index, from_kwh / B_);
    }

    bool leg_ok(int a, int b, double soc) const {
        double e = E_[a * N_ + b];
        return std::isfinite(e) && margin_ * e <= soc + kEps;
    }

    // ---- precomputed fronts --------------------------------------------------
    // fronts_[state * (nc_ + 1) + target], state = station_index * nl + level.
    void build_fronts() {
        int nl = static_cast<int>(levels_.size());
        fronts_.assign(static_cast<size_t>(ns_) * nl * (nc_ + 1), {});
        path_off_.clear();
        path_len_.clear();
        path_pool_.clear();
        for (int s = 0; s < ns_; ++s) {
            for (int l = 0; l < nl; ++l) search_from(s, l);
        }
    }

    struct SLabel {
        int node;  // station node index
        double cost, time, soc;
        int parent;
        int level;  // -1 pass, >= 0 charged
        bool dead;
    };

    void search_from(int s0, int l0) {
        int nl = static_cast<int>(levels_.size());
        std::vector<SLabel> labels;
        std::vector<std::vector<int>> at(ns_);  // label ids per station (alive)
        std::vector<int> queue;
        labels.push_back({nc_ + 1 + s0, 0.0, 0.0, levels_[l0] * B_, -1, l0, false});
        at[s0].push_back(0);
        queue.push_back(0);
        auto& out = fronts_;
        size_t base = static_cast<size_t>(s0 * nl + l0) * (nc_ + 1);
        std::vector<std::vector<std::pair<Lab, int>>> terminal(nc_ + 1);  // pareto + label id

        auto insert_label = [&](int station, SLabel cand) {
            auto& list = at[station];
            Lab c{cand.cost, cand.time, cand.soc};
            for (int id : list) {
                const SLabel& o = labels[id];
                if (dominates(Lab{o.cost, o.time, o.soc}, c)) return;
            }
            for (int id : list) {
                SLabel& o = labels[id];
                if (dominates(c, Lab{o.cost, o.time, o.soc})) o.dead = true;
            }
            list.erase(std::remove_if(list.begin(), list.end(), [&](int id) { return labels[id].dead; }),
                       list.end());
            if (static_cast<int>(list.size()) >= max_labels_) {
                ++cap_hits_;
                return;
            }
            labels.push_back(cand);
            int id = static_cast<int>(labels.size()) - 1;
            list.push_back(id);
            queue.push_back(id);
        };

        for (size_t head = 0; head < queue.size(); ++head) {
            int id = queue[head];
            if (labels[id].dead) continue;
            SLabel cur = labels[id];
            int a = cur.node;
            // terminals
            for (int j = 0; j <= nc_; ++j) {
                if (!leg_ok(a, j, cur.soc)) continue;
                double hv = H_[a * N_ + j];
                Lab cand{cur.cost + hv, cur.time + hv, cur.soc - E_[a * N_ + j]};
                if (cand.time > horizon_ + kEps) continue;
                auto& tl = terminal[j];
                bool skip = false;
                for (auto& t : tl) {
                    if (dominates(t.first, cand)) { skip = true; break; }
                }
                if (skip) continue;
                tl.erase(std::remove_if(tl.begin(), tl.end(),
                                        [&](const std::pair<Lab, int>& t) { return dominates(cand, t.first); }),
                         tl.end());
                tl.emplace_back(cand, id);
            }
            // stations
            for (int s = 0; s < ns_; ++s) {
                int b = nc_ + 1 + s;
                if (b == a || !leg_ok(a, b, cur.soc)) continue;
                double hv = H_[a * N_ + b];
                double arr = cur.soc - E_[a * N_ + b];
                double t1 = cur.time + hv;
                if (t1 > horizon_ + kEps) continue;
                insert_label(s, {b, cur.cost + hv, t1, arr, id, -1, false});
                for (int l = 0; l < nl; ++l) {
                    if (levels_[l] * B_ <= arr + kEps) continue;
                    double dF = charge_hours(s, arr, levels_[l]);
                    if (t1 + dF > horizon_ + kEps) continue;
                    insert_label(s, {b, cur.cost + hv + w_ * dF, t1 + dF, levels_[l] * B_, id, l, false});
                }
            }
        }
        for (int j = 0; j <= nc_; ++j) {
            auto& dest = out[base + j];
            for (auto& t : terminal[j]) {
                // reconstruct station chain (excluding the start state itself)
                std::vector<Stop> chain;
                int id = t.second;
                while (labels[id].parent >= 0) {
                    chain.push_back({labels[id].node, labels[id].level});
                    id = labels[id].parent;
                }
                std::reverse(chain.begin(), chain.end());
                int path = -1;
                if (!chain.empty()) {
                    path = static_cast<int>(path_off_.size());
                    path_off_.push_back(static_cast<int>(path_pool_.size()));
                    path_len_.push_back(static_cast<int>(chain.size()));
                    for (auto& st : chain) path_pool_.push_back(st);
                }
                dest.push_back({t.first.cost, t.first.time, t.first.soc, path});
            }
        }
    }

    // ---- route DP --------------------------------------------------------------
    struct DLabel {
        Lab lab;       // departure label (after service) / arrival label at the depot
        int parent;    // index in previous position's list
        int via_station;  // -1 direct, else station index (0..ns_-1)
        int via_level;    // -1 pass-through, >= 0 charge level (then `path` follows)
        int path;         // path id in the table, -1 none
    };

    // Returns the best final label index or -1. `layers` is filled for reconstruction.
    bool run_dp(const std::vector<int>& seq, std::vector<std::vector<DLabel>>& layers) const {
        int m = static_cast<int>(seq.size());
        std::vector<int> route;
        route.reserve(m + 2);
        route.push_back(0);
        for (int c : seq) route.push_back(c);
        route.push_back(0);
        layers.assign(m + 2, {});
        layers[0].push_back({{0.0, 0.0, x0_}, -1, -1, -1, -1});
        int nl = static_cast<int>(levels_.size());

        for (int p = 0; p + 1 < static_cast<int>(route.size()); ++p) {
            int i = route[p], j = route[p + 1];
            bool last = (p + 1 == static_cast<int>(route.size()) - 1);
            auto& next = layers[p + 1];
            const auto& cur = layers[p];
            for (int di = 0; di < static_cast<int>(cur.size()); ++di) {
                const Lab D = cur[di].lab;
                auto consider = [&](double cost, double arr_time, double soc, int st, int lv, int path) {
                    double t_ready, c_after;
                    if (last) {
                        if (arr_time > horizon_ + kEps) return;
                        t_ready = arr_time;
                        c_after = cost;
                    } else {
                        if (arr_time > close_[j] + kEps) return;
                        double start = std::max(arr_time, open_[j]);
                        t_ready = start + service_[j];
                        if (t_ready > horizon_ + kEps) return;
                        c_after = cost + w_ * service_[j];
                    }
                    Lab cand{c_after, t_ready, soc};
                    for (const auto& o : next) {
                        if (dominates(o.lab, cand)) return;
                    }
                    next.erase(std::remove_if(next.begin(), next.end(),
                                              [&](const DLabel& o) { return dominates(cand, o.lab); }),
                               next.end());
                    next.push_back({cand, di, st, lv, path});
                };
                // direct
                if (leg_ok(i, j, D.soc)) {
                    double hv = H_[i * N_ + j];
                    consider(D.cost + hv, D.time + hv, D.soc - E_[i * N_ + j], -1, -1, -1);
                }
                for (int s = 0; s < ns_; ++s) {
                    int b = nc_ + 1 + s;
                    if (!leg_ok(i, b, D.soc)) continue;
                    double hv = H_[i * N_ + b];
                    double arr = D.soc - E_[i * N_ + b];
                    double c1 = D.cost + hv, t1 = D.time + hv;
                    if (t1 > horizon_ + kEps) continue;
                    if (leg_ok(b, j, arr)) {  // pass through the charger
                        double h2 = H_[b * N_ + j];
                        consider(c1 + h2, t1 + h2, arr - E_[b * N_ + j], s, -1, -1);
                    }
                    for (int l = 0; l < nl; ++l) {
                        if (levels_[l] * B_ <= arr + kEps) continue;
                        double dF = charge_hours(s, arr, levels_[l]);
                        double c2 = c1 + w_ * dF, t2 = t1 + dF;
                        if (t2 > horizon_ + kEps) continue;
                        const auto& front = fronts_[(static_cast<size_t>(s) * nl + l) * (nc_ + 1) + j];
                        for (const auto& q : front) {
                            consider(c2 + q.cost, t2 + q.time, q.soc, s, l, q.path);
                        }
                    }
                }
            }
            if (next.empty()) return false;
        }
        return true;
    }

    double price_cost(const std::vector<int>& seq) const {
        std::vector<std::vector<DLabel>> layers;
        if (!feasible_load(seq) || !run_dp(seq, layers)) return kInf;
        double best = kInf;
        for (const auto& d : layers.back()) best = std::min(best, d.lab.cost);
        return best;
    }

    std::vector<int> layer_sizes(const std::vector<int>& seq) const {
        std::vector<std::vector<DLabel>> layers;
        run_dp(seq, layers);
        std::vector<int> sizes;
        for (auto& l : layers) sizes.push_back(static_cast<int>(l.size()));
        return sizes;
    }

    bool feasible_load(const std::vector<int>& seq) const {
        double load = 0.0;
        for (int c : seq) load += demand_[c];
        return load <= payload_ + kEps;
    }

    // Returns (cost, time, stops) or None. stops = [(node, level)], depot first and last.
    py::object price_route(const std::vector<int>& seq) const {
        std::vector<std::vector<DLabel>> layers;
        if (seq.empty() || !feasible_load(seq) || !run_dp(seq, layers)) return py::none();
        int best = -1;
        for (int k = 0; k < static_cast<int>(layers.back().size()); ++k) {
            const auto& d = layers.back()[k];
            if (best < 0 || d.lab.cost < layers.back()[best].lab.cost - 1e-12 ||
                (std::fabs(d.lab.cost - layers.back()[best].lab.cost) <= 1e-12 &&
                 d.lab.time < layers.back()[best].lab.time)) {
                best = k;
            }
        }
        int m = static_cast<int>(seq.size());
        std::vector<int> route{0};
        for (int c : seq) route.push_back(c);
        route.push_back(0);
        // walk back collecting per-leg stop chains
        std::vector<std::vector<Stop>> legs(m + 1);
        int idx = best;
        for (int p = m + 1; p >= 1; --p) {
            const DLabel& d = layers[p][idx];
            std::vector<Stop>& leg = legs[p - 1];
            if (d.via_station >= 0) {
                leg.push_back({nc_ + 1 + d.via_station, d.via_level});
                if (d.path >= 0) {
                    for (int q = 0; q < path_len_[d.path]; ++q) leg.push_back(path_pool_[path_off_[d.path] + q]);
                }
            }
            idx = d.parent;
        }
        std::vector<Stop> stops;
        for (int p = 0; p <= m + 1; ++p) {
            stops.push_back({route[p], -2});
            if (p <= m) for (auto& s : legs[p]) stops.push_back(s);
        }
        std::vector<std::pair<int, int>> flat;
        for (auto& s : stops) flat.emplace_back(s.node, s.level);
        return py::make_tuple(layers.back()[best].lab.cost, layers.back()[best].lab.time, flat);
    }

    std::vector<double> price_routes(const std::vector<std::vector<int>>& seqs) const {
        std::vector<double> out(seqs.size());
#pragma omp parallel for schedule(dynamic) if (seqs.size() > 64)
        for (int k = 0; k < static_cast<int>(seqs.size()); ++k) out[k] = price_cost(seqs[k]);
        return out;
    }

    // Re-simulate a stop list. Returns list of
    // (node, level, arrive_time, arrive_soc_kwh, depart_time, depart_soc_kwh, charge_hours)
    // or None if infeasible. Independent of the DP: used as the checker.
    py::object simulate(const std::vector<std::pair<int, int>>& stops) const {
        std::vector<py::tuple> rows;
        double t = 0.0, soc = x0_, cost = 0.0, load = 0.0;
        for (size_t k = 0; k < stops.size(); ++k) {
            int node = stops[k].first, level = stops[k].second;
            double arrive_t = t, arrive_soc = soc, charge_h = 0.0;
            if (k > 0) {
                int prev = stops[k - 1].first;
                if (!leg_ok(prev, node, soc)) return py::none();
                double hv = H_[prev * N_ + node];
                arrive_t = t + hv;
                arrive_soc = soc - E_[prev * N_ + node];
                cost += hv;
            }
            double dep_t = arrive_t, dep_soc = arrive_soc;
            bool is_station = node > nc_;
            bool is_last = (k + 1 == stops.size());
            if (is_station) {
                if (level >= 0) {
                    if (levels_[level] * B_ <= arrive_soc + kEps) return py::none();
                    charge_h = charge_hours(node - nc_ - 1, arrive_soc, levels_[level]);
                    dep_t += charge_h;
                    dep_soc = levels_[level] * B_;
                    cost += w_ * charge_h;
                }
            } else if (node > 0) {
                if (arrive_t > close_[node] + kEps) return py::none();
                dep_t = std::max(arrive_t, open_[node]) + service_[node];
                load += demand_[node];
                cost += w_ * service_[node];
                if (load > payload_ + kEps) return py::none();
            }
            if (is_last) {
                if (arrive_t > horizon_ + kEps) return py::none();
            } else if (dep_t > horizon_ + kEps) {
                return py::none();
            }
            rows.push_back(py::make_tuple(node, level, arrive_t, arrive_soc, dep_t, dep_soc, charge_h));
            t = dep_t;
            soc = dep_soc;
        }
        return py::make_tuple(cost, py::cast(rows));
    }

    // Optimistic per-leg table for the outer search: for every depot/customer
    // pair, the cheapest cost and the earliest time over all station detours,
    // starting from `soc_fraction` of the battery. Both are lower bounds of what
    // the same leg costs inside a real route when soc_fraction = 1.
    std::pair<py::array_t<double>, py::array_t<double>> leg_surrogate(double soc_fraction) const {
        int n = nc_ + 1;
        py::array_t<double> cost({n, n}), time({n, n});
        auto c = cost.mutable_unchecked<2>();
        auto t = time.mutable_unchecked<2>();
        int nl = static_cast<int>(levels_.size());
        double x = soc_fraction * B_;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) { c(i, j) = kInf; t(i, j) = kInf; }
            auto upd = [&](int j, double cc, double tt) {
                if (j == i) return;
                c(i, j) = std::min(c(i, j), cc);
                t(i, j) = std::min(t(i, j), tt);
            };
            for (int j = 0; j < n; ++j) {
                if (j != i && leg_ok(i, j, x)) upd(j, H_[i * N_ + j], H_[i * N_ + j]);
            }
            for (int s = 0; s < ns_; ++s) {
                int b = nc_ + 1 + s;
                if (!leg_ok(i, b, x)) continue;
                double hv = H_[i * N_ + b], arr = x - E_[i * N_ + b];
                for (int j = 0; j < n; ++j) {
                    if (j != i && leg_ok(b, j, arr)) upd(j, hv + H_[b * N_ + j], hv + H_[b * N_ + j]);
                }
                for (int l = 0; l < nl; ++l) {
                    if (levels_[l] * B_ <= arr + kEps) continue;
                    double dF = charge_hours(s, arr, levels_[l]);
                    const size_t off = (static_cast<size_t>(s) * nl + l) * (nc_ + 1);
                    for (int j = 0; j < n; ++j) {
                        if (j == i) continue;
                        for (const auto& q : fronts_[off + j]) {
                            upd(j, hv + w_ * dF + q.cost, hv + dF + q.time);
                        }
                    }
                }
            }
        }
        return {cost, time};
    }

private:
    int nc_, N_ = 0, ns_ = 0;
    double B_, x0_, margin_, horizon_, payload_;
    std::vector<double> levels_;
    std::vector<std::vector<double>> curves_;
    std::vector<int> curve_of_;
    double step_, w_;
    int max_labels_;
    mutable int cap_hits_ = 0;
    std::vector<double> demand_, service_, open_, close_;
    std::vector<double> E_, H_;
    std::vector<std::vector<FrontPt>> fronts_;
    std::vector<int> path_off_, path_len_;
    std::vector<Stop> path_pool_;
};

}  // namespace

PYBIND11_MODULE(evrp_core, m) {
    m.doc() = "Exact route pricing for the nominal EVRP (RoutingBlocks/frvcpy-style baseline)";
    py::class_<Core>(m, "Core")
        .def(py::init<py::array_t<double>, py::array_t<double>, int, std::vector<double>,
                      std::vector<double>, std::vector<double>, std::vector<double>, double, double,
                      double, double, double, std::vector<double>,
                      std::vector<std::vector<double>>, std::vector<int>, double, double, int>(),
             py::arg("energy"), py::arg("hours"), py::arg("n_customers"), py::arg("demand"),
             py::arg("service"), py::arg("tw_open"), py::arg("tw_close"), py::arg("battery"),
             py::arg("start_soc_kwh"), py::arg("margin"), py::arg("horizon"), py::arg("payload"),
             py::arg("levels"), py::arg("curves"), py::arg("curve_of_station"),
             py::arg("curve_step"), py::arg("service_cost_weight") = 0.0,
             py::arg("max_labels") = 512)
        .def("price_route", &Core::price_route, py::arg("customers"))
        .def("price_routes", &Core::price_routes, py::arg("routes"), py::call_guard<py::gil_scoped_release>())
        .def("simulate", &Core::simulate, py::arg("stops"))
        .def("layer_sizes", &Core::layer_sizes, py::arg("customers"))
        .def("leg_surrogate", &Core::leg_surrogate, py::arg("soc_fraction") = 1.0)
        .def("front_points", &Core::front_points)
        .def("label_cap_hits", &Core::label_cap_hits)
        .def_property_readonly("n_nodes", &Core::n_nodes)
        .def_property_readonly("n_stations", &Core::n_stations);
}
