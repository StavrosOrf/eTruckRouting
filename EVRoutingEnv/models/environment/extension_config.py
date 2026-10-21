"""Opt-in configuration blocks for the dynamic and stochastic model components.

Every block is off when absent, so a legacy configuration produces the frozen
golden trajectories. Values are validated here, once, and resolved presets are
written back into the configuration so that artifact hashes record them. An
unsupported combination raises ``ValueError`` with the reason; nothing is
silently relaxed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real

from EVRoutingEnv.models.core.hours_of_service import REGIMES, HosRegime


KULLMAN_FAST_MINUTES = 26.62  # Kullman et al. (2021) sec. 7.1, after Morrissey et al. (2016)
KULLMAN_MODERATE_MINUTES = 128.78
LOOKUP_DCFAST_MINUTES = 30.0  # waiting_time_lookup.json, DCFast
REMPEL_AVAILABILITY = 0.733  # Rempel et al. (2023): 73.3% of 655 DC-fast ports functional
HOS_MAX_TIME_HOURS = 96.0


def _bool(value, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be boolean")
    return value


def _choice(value, options, label: str) -> str:
    if value not in options:
        raise ValueError(f"{label} must be one of {sorted(options)}, got {value!r}")
    return value


def _finite(value, label: str, *, low: float = 0.0, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < low or (strict and result <= low):
        bound = ">" if strict else ">="
        raise ValueError(f"{label} must be finite and {bound} {low}")
    return result


def _integer(value, label: str, *, low: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < low:
        raise ValueError(f"{label} must be an integer >= {low}")
    return int(value)


def _unknown_keys(block: dict, allowed: set[str], label: str) -> None:
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} keys: {unknown}")


@dataclass(frozen=True)
class OutageConfig:
    enabled: bool = False
    port_availability: float = REMPEL_AVAILABILITY
    mean_repair_hours: float = 24.0
    interrupt_sessions: bool = False

    @property
    def mean_up_hours(self) -> float:
        # MTBF = MTTR * a / (1 - a)
        a = self.port_availability
        return math.inf if a >= 1.0 else self.mean_repair_hours * a / (1.0 - a)


@dataclass(frozen=True)
class PublicChargingConfig:
    enabled: bool = False
    queue_model: str = "dynamic_mmc"
    queue_policy: str = "kullman2021"
    information: str = "on_arrival"
    memory: str = "shared"
    max_stations_between_stops: int | None = None
    arrival_model: str = "homogeneous_poisson"
    utilization: float = 0.65
    mean_service_hours: float = KULLMAN_FAST_MINUTES / 60.0
    service_preset: str | None = "kullman_fast"
    hourly_profile: tuple[float, ...] | None = None
    initial_state: str = "stationary"
    outages: OutageConfig = field(default_factory=OutageConfig)
    debug_eager_public_queues: bool = False


@dataclass(frozen=True)
class DepotChargingConfig:
    enabled: bool = False
    ports: int | None = None
    power_kw: float = 150.0
    parallel_with_loading: bool = False


@dataclass(frozen=True)
class HoursOfServiceConfig:
    enabled: bool = False
    regime: str = "us_fmcsa_property"
    rest_locations: str = "anywhere"
    initial_state: str = "fresh"
    charging_counts_as_break: bool = True
    queue_wait_counts_as_break: bool = False
    handling_overhead_hours: float = 5.0 / 60.0
    eu_split_break: bool = True
    eu_extended_daily_driving: bool = False
    eu_reduced_daily_rest: bool = False
    time_guard: str = "realized"

    def regime_model(self) -> HosRegime:
        return HosRegime.build(
            self.regime,
            charging_counts_as_break=self.charging_counts_as_break,
            queue_wait_counts_as_break=self.queue_wait_counts_as_break,
            eu_split_break=self.eu_split_break,
            eu_extended_daily_driving=self.eu_extended_daily_driving,
            eu_reduced_daily_rest=self.eu_reduced_daily_rest,
        )


@dataclass(frozen=True)
class TerminalConfig:
    name: str
    lat: float
    lon: float


@dataclass(frozen=True)
class DrayageConfig:
    terminals: tuple[TerminalConfig, ...] = ()
    import_share: float = 0.5
    origin_service_hours: float = 2.0 / 60.0
    destination_service_hours: float = 4.0 / 60.0
    delay_penalty_per_hour: float | None = None
    energy_model: str = "network"
    max_terminal_distance_km: float = 5.0
    container_mass_kg: tuple[float, float] = (10_000.0, 20_000.0)


@dataclass(frozen=True)
class RequestsConfig:
    mode: str = "static"
    arrival_process: str = "euro_neurips_epochs"
    epoch_hours: float = 1.0
    dispatch_margin_hours: float = 1.0
    max_requests_per_epoch: int = 100
    poisson_rate_per_hour: float | None = None
    initial_requests: int = 0
    request_cutoff_hours: float | None = None
    max_requests: int = 200
    static_context_size: int = 230
    time_windows: dict = field(default_factory=dict)
    deadline_type: str = "hard"
    allow_preemptive_returns: bool = False
    loading_service_hours: float = 0.2
    lost_request_penalty_hours: float = 24.0
    wait_hours: float = 0.5

    @property
    def dynamic(self) -> bool:
        return self.mode != "static"


@dataclass(frozen=True)
class ExtensionConfig:
    public_charging: PublicChargingConfig
    depot_charging: DepotChargingConfig
    hours_of_service: HoursOfServiceConfig
    requests: RequestsConfig
    drayage: DrayageConfig | None
    in_flight_information: str
    loading: str

    @property
    def any_enabled(self) -> bool:
        return (
            self.public_charging.enabled
            or self.depot_charging.enabled
            or self.hours_of_service.enabled
            or self.requests.dynamic
            or self.in_flight_information != "realized"
            or self.loading != "implicit"
        )


def _parse_public(block: dict | None, charging_config: dict) -> PublicChargingConfig:
    if block is None:
        return PublicChargingConfig()
    block = dict(block)
    _unknown_keys(block, {
        "enabled", "queue_model", "queue_policy", "information", "memory",
        "max_stations_between_stops", "exogenous_traffic", "outages",
        "debug_eager_public_queues",
    }, "public_charging")
    enabled = _bool(block.get("enabled", False), "public_charging.enabled")
    queue_model = _choice(
        block.get("queue_model", "dynamic_mmc"),
        {"dynamic_mmc", "iid_stationary_wait"}, "public_charging.queue_model",
    )
    queue_policy = _choice(
        block.get("queue_policy", "kullman2021"),
        {"kullman2021", "legacy"}, "public_charging.queue_policy",
    )
    information = _choice(
        block.get("information", "on_arrival"),
        {"on_arrival", "realtime_queue", "occupancy_indicator"},
        "public_charging.information",
    )
    memory = _choice(block.get("memory", "shared"), {"shared", "per_truck"}, "public_charging.memory")
    between = block.get("max_stations_between_stops")
    if between is not None:
        between = _integer(between, "public_charging.max_stations_between_stops", low=1)
    if queue_policy == "kullman2021" and queue_model != "dynamic_mmc":
        raise ValueError(
            "public_charging.queue_policy=kullman2021 requires queue_model=dynamic_mmc: "
            "the queue-or-leave decision needs an observable queue"
        )
    if queue_model == "iid_stationary_wait" and information != "on_arrival":
        raise ValueError(
            "public_charging.queue_model=iid_stationary_wait has no queue state to "
            "reveal; it requires information=on_arrival"
        )

    traffic = dict(block.get("exogenous_traffic") or {})
    _unknown_keys(traffic, {
        "arrival_model", "utilization", "service_preset", "mean_service_hours",
        "hourly_profile", "initial_state", "outside_battery_kwh",
        "resolved_mean_service_hours",
    }, "public_charging.exogenous_traffic")
    arrival_model = _choice(
        traffic.get("arrival_model", "homogeneous_poisson"),
        {"homogeneous_poisson", "piecewise_poisson"},
        "public_charging.exogenous_traffic.arrival_model",
    )
    utilization = _finite(
        traffic.get("utilization", 0.65),
        "public_charging.exogenous_traffic.utilization", strict=True,
    )
    if utilization >= 1.0:
        raise ValueError(
            "public_charging.exogenous_traffic.utilization must be < 1: an M/M/c queue "
            "with u >= 1 has no stationary regime"
        )
    preset = traffic.get("service_preset", "kullman_fast" if "mean_service_hours" not in traffic else None)
    explicit = traffic.get("mean_service_hours")
    if (preset is None) == (explicit is None):
        raise ValueError(
            "public_charging.exogenous_traffic: give exactly one of service_preset "
            "and mean_service_hours"
        )
    if explicit is not None:
        mean_service = _finite(explicit, "public_charging.exogenous_traffic.mean_service_hours", strict=True)
    else:
        mean_service = _service_preset_hours(preset, traffic, charging_config)
    resolved = traffic.get("resolved_mean_service_hours")
    if resolved is not None and abs(float(resolved) - mean_service) > 1e-12:
        raise ValueError(
            "exogenous_traffic.resolved_mean_service_hours does not match the preset; "
            "it is written by the environment and must not be edited"
        )
    profile = traffic.get("hourly_profile")
    if arrival_model == "piecewise_poisson":
        if profile is None or len(profile) != 24:
            raise ValueError(
                "piecewise_poisson requires exogenous_traffic.hourly_profile with 24 values"
            )
        profile = tuple(
            _finite(value, "public_charging.exogenous_traffic.hourly_profile") for value in profile
        )
        if abs(sum(profile) / 24.0 - 1.0) > 1e-9:
            raise ValueError("exogenous_traffic.hourly_profile must have mean 1 (within 1e-9)")
    elif profile is not None:
        raise ValueError("hourly_profile applies only to arrival_model=piecewise_poisson")
    initial_state = _choice(
        traffic.get("initial_state", "stationary"), {"stationary", "empty"},
        "public_charging.exogenous_traffic.initial_state",
    )

    outages_block = dict(block.get("outages") or {})
    _unknown_keys(outages_block, {
        "enabled", "port_availability", "mean_repair_hours", "interrupt_sessions",
    }, "public_charging.outages")
    availability = _finite(
        outages_block.get("port_availability", REMPEL_AVAILABILITY),
        "public_charging.outages.port_availability", strict=True,
    )
    if availability > 1.0:
        raise ValueError("public_charging.outages.port_availability must be in (0, 1]")
    interrupt = _bool(
        outages_block.get("interrupt_sessions", False), "public_charging.outages.interrupt_sessions"
    )
    if interrupt:
        raise ValueError(
            "public_charging.outages.interrupt_sessions=true is not modelled: a failing "
            "port finishes its session (Section 5.2.9)"
        )
    outages = OutageConfig(
        enabled=_bool(outages_block.get("enabled", False), "public_charging.outages.enabled"),
        port_availability=availability,
        mean_repair_hours=_finite(
            outages_block.get("mean_repair_hours", 24.0),
            "public_charging.outages.mean_repair_hours", strict=True,
        ),
        interrupt_sessions=interrupt,
    )
    return PublicChargingConfig(
        enabled=enabled,
        queue_model=queue_model,
        queue_policy=queue_policy,
        information=information,
        memory=memory,
        max_stations_between_stops=between,
        arrival_model=arrival_model,
        utilization=utilization,
        mean_service_hours=mean_service,
        service_preset=preset,
        hourly_profile=profile,
        initial_state=initial_state,
        outages=outages,
        debug_eager_public_queues=_bool(
            block.get("debug_eager_public_queues", False),
            "public_charging.debug_eager_public_queues",
        ),
    )


def _service_preset_hours(preset, traffic: dict, charging_config: dict) -> float:
    if preset == "kullman_fast":
        return KULLMAN_FAST_MINUTES / 60.0
    if preset == "kullman_moderate":
        return KULLMAN_MODERATE_MINUTES / 60.0
    if preset == "lookup_dcfast":
        return LOOKUP_DCFAST_MINUTES / 60.0
    if preset == "keskin_truck":
        # Keskin et al. (2019, WSC) sec. 2.2: E[recharge] = 0.55 Q g, i.e. 55% of
        # a battery Q at g hours per kWh, applied to outside trucks.
        battery = _finite(
            traffic.get("outside_battery_kwh", 400.0),
            "public_charging.exogenous_traffic.outside_battery_kwh", strict=True,
        )
        dcfast = charging_config.get("dcfast", {})
        power = float(dcfast.get("charge_rate", 150.0))
        efficiency = float(dcfast.get("efficiency", 0.9))
        return 0.55 * battery / (power * efficiency)
    raise ValueError(
        "public_charging.exogenous_traffic.service_preset must be one of "
        "['keskin_truck', 'kullman_fast', 'kullman_moderate', 'lookup_dcfast']"
    )


def _parse_depot(block: dict | None) -> DepotChargingConfig:
    if block is None:
        return DepotChargingConfig()
    block = dict(block)
    _unknown_keys(block, {"enabled", "ports", "power_kw", "parallel_with_loading"}, "depot_charging")
    ports = block.get("ports")
    if ports is not None:
        ports = _integer(ports, "depot_charging.ports", low=1)
    parallel = _bool(block.get("parallel_with_loading", False), "depot_charging.parallel_with_loading")
    if parallel:
        raise ValueError(
            "depot_charging.parallel_with_loading=true is not modelled; loading and "
            "depot charging are sequential (case X2)"
        )
    return DepotChargingConfig(
        enabled=_bool(block.get("enabled", False), "depot_charging.enabled"),
        ports=ports,
        power_kw=_finite(block.get("power_kw", 150.0), "depot_charging.power_kw", strict=True),
        parallel_with_loading=parallel,
    )


def _parse_hos(block: dict | None, max_time: float) -> HoursOfServiceConfig:
    if block is None:
        return HoursOfServiceConfig()
    block = dict(block)
    _unknown_keys(block, {
        "enabled", "regime", "rest_locations", "initial_state", "charging_counts_as_break",
        "queue_wait_counts_as_break", "handling_overhead_hours", "eu_split_break",
        "eu_extended_daily_driving", "eu_reduced_daily_rest", "time_guard",
    }, "hours_of_service")
    enabled = _bool(block.get("enabled", False), "hours_of_service.enabled")
    regime = _choice(block.get("regime", "us_fmcsa_property"), set(REGIMES), "hours_of_service.regime")
    rest_locations = _choice(
        block.get("rest_locations", "anywhere"), {"anywhere", "stops_only"},
        "hours_of_service.rest_locations",
    )
    if regime == "bai2023_simplified" and rest_locations != "stops_only":
        raise ValueError("hours_of_service.regime=bai2023_simplified requires rest_locations=stops_only")
    initial = _choice(block.get("initial_state", "fresh"), {"fresh"}, "hours_of_service.initial_state")
    if enabled and max_time > HOS_MAX_TIME_HOURS + 1e-9:
        raise ValueError(
            f"hours_of_service requires environment.max_time <= {HOS_MAX_TIME_HOURS:g}: "
            "weekly limits (US 60/70 h, EU 56/90 h driving and weekly rest) are not "
            "modelled, so longer horizons could let them bind"
        )
    return HoursOfServiceConfig(
        enabled=enabled,
        regime=regime,
        rest_locations=rest_locations,
        initial_state=initial,
        charging_counts_as_break=_bool(
            block.get("charging_counts_as_break", True), "hours_of_service.charging_counts_as_break"
        ),
        queue_wait_counts_as_break=_bool(
            block.get("queue_wait_counts_as_break", False),
            "hours_of_service.queue_wait_counts_as_break",
        ),
        handling_overhead_hours=_finite(
            block.get("handling_overhead_hours", 5.0 / 60.0),
            "hours_of_service.handling_overhead_hours",
        ),
        eu_split_break=_bool(block.get("eu_split_break", True), "hours_of_service.eu_split_break"),
        eu_extended_daily_driving=_bool(
            block.get("eu_extended_daily_driving", False),
            "hours_of_service.eu_extended_daily_driving",
        ),
        eu_reduced_daily_rest=_bool(
            block.get("eu_reduced_daily_rest", False), "hours_of_service.eu_reduced_daily_rest"
        ),
        time_guard=_choice(
            block.get("time_guard", "realized"), {"realized", "worst_case"},
            "hours_of_service.time_guard",
        ),
    )


def _parse_requests(block: dict | None, max_time: float) -> RequestsConfig:
    if block is None:
        return RequestsConfig()
    block = dict(block)
    _unknown_keys(block, {
        "mode", "arrival_process", "epoch_hours", "dispatch_margin_hours",
        "max_requests_per_epoch", "poisson_rate_per_hour", "initial_requests",
        "request_cutoff_hours", "max_requests", "static_context_size", "time_windows",
        "deadline_type", "allow_preemptive_returns", "loading_service_hours",
        "lost_request_penalty_hours", "wait_hours",
    }, "requests")
    mode = _choice(block.get("mode", "static"), {"static", "same_day", "drayage"}, "requests.mode")
    arrival = _choice(
        block.get("arrival_process", "euro_neurips_epochs"),
        {"euro_neurips_epochs", "poisson"}, "requests.arrival_process",
    )
    rate = block.get("poisson_rate_per_hour")
    if arrival == "poisson":
        if rate is None:
            raise ValueError("requests.arrival_process=poisson requires poisson_rate_per_hour")
        rate = _finite(rate, "requests.poisson_rate_per_hour", strict=True)
    elif rate is not None:
        raise ValueError("requests.poisson_rate_per_hour applies only to arrival_process=poisson")
    margin = _finite(block.get("dispatch_margin_hours", 1.0), "requests.dispatch_margin_hours")
    cutoff = block.get("request_cutoff_hours")
    cutoff = max_time - margin if cutoff is None else _finite(cutoff, "requests.request_cutoff_hours")
    return RequestsConfig(
        mode=mode,
        arrival_process=arrival,
        epoch_hours=_finite(block.get("epoch_hours", 1.0), "requests.epoch_hours", strict=True),
        dispatch_margin_hours=margin,
        max_requests_per_epoch=_integer(
            block.get("max_requests_per_epoch", 100), "requests.max_requests_per_epoch", low=1
        ),
        poisson_rate_per_hour=rate,
        initial_requests=_integer(block.get("initial_requests", 0), "requests.initial_requests"),
        request_cutoff_hours=cutoff,
        max_requests=_integer(block.get("max_requests", 200), "requests.max_requests", low=1),
        static_context_size=_integer(
            block.get("static_context_size", 230), "requests.static_context_size", low=1
        ),
        time_windows=dict(block.get("time_windows") or {}),
        deadline_type=_choice(block.get("deadline_type", "hard"), {"hard", "soft"}, "requests.deadline_type"),
        allow_preemptive_returns=_bool(
            block.get("allow_preemptive_returns", False), "requests.allow_preemptive_returns"
        ),
        loading_service_hours=_finite(
            block.get("loading_service_hours", 0.2), "requests.loading_service_hours"
        ),
        lost_request_penalty_hours=_finite(
            block.get("lost_request_penalty_hours", 24.0), "requests.lost_request_penalty_hours"
        ),
        wait_hours=_finite(block.get("wait_hours", 0.5), "requests.wait_hours", strict=True),
    )


def _parse_drayage(block: dict | None) -> DrayageConfig | None:
    if block is None:
        return None
    block = dict(block)
    _unknown_keys(block, {
        "terminals", "import_share", "origin_service_hours", "destination_service_hours",
        "delay_penalty_per_hour", "energy_model", "max_terminal_distance_km", "container_mass_kg",
    }, "drayage")
    terminals = []
    for index, item in enumerate(block.get("terminals") or ()):
        terminals.append(TerminalConfig(
            name=str(item["name"]),
            lat=_finite(item["lat"], f"drayage.terminals[{index}].lat", low=-90.0),
            lon=_finite(item["lon"], f"drayage.terminals[{index}].lon", low=-180.0),
        ))
    if not terminals:
        raise ValueError("drayage.terminals must list at least one terminal")
    share = _finite(block.get("import_share", 0.5), "drayage.import_share")
    if share > 1.0:
        raise ValueError("drayage.import_share must be in [0, 1]")
    penalty = block.get("delay_penalty_per_hour")
    if penalty is not None:
        penalty = _finite(penalty, "drayage.delay_penalty_per_hour")
    mass = tuple(block.get("container_mass_kg", (10_000.0, 20_000.0)))
    if len(mass) != 2 or not 0 < float(mass[0]) <= float(mass[1]):
        raise ValueError("drayage.container_mass_kg must be [min, max] with 0 < min <= max")
    return DrayageConfig(
        terminals=tuple(terminals),
        import_share=share,
        origin_service_hours=_finite(
            block.get("origin_service_hours", 2.0 / 60.0), "drayage.origin_service_hours"
        ),
        destination_service_hours=_finite(
            block.get("destination_service_hours", 4.0 / 60.0), "drayage.destination_service_hours"
        ),
        delay_penalty_per_hour=penalty,
        energy_model=_choice(
            block.get("energy_model", "network"), {"network", "kancharla_ramadurai_2020"},
            "drayage.energy_model",
        ),
        max_terminal_distance_km=_finite(
            block.get("max_terminal_distance_km", 5.0), "drayage.max_terminal_distance_km", strict=True
        ),
        container_mass_kg=(float(mass[0]), float(mass[1])),
    )


def parse_extension_config(config: dict, *, max_time: float, sequential: bool,
                           preassigned: bool) -> ExtensionConfig:
    """Validate every opt-in block and the cross-block combinations."""
    problem = config.get("problem", {}) or {}
    observation = dict(config.get("observation") or {})
    _unknown_keys(observation, {"in_flight_information"}, "observation")
    in_flight = _choice(
        observation.get("in_flight_information", "realized"),
        {"realized", "nominal_eta"}, "observation.in_flight_information",
    )
    loading = _choice(problem.get("loading", "implicit"), {"implicit", "at_depot"}, "problem.loading")

    public = _parse_public(config.get("public_charging"), config.get("charging", {}))
    depot = _parse_depot(config.get("depot_charging"))
    hos = _parse_hos(config.get("hours_of_service"), max_time)
    requests = _parse_requests(config.get("requests"), max_time)
    drayage = _parse_drayage(config.get("drayage"))

    dynamic = requests.dynamic or loading == "at_depot"
    if dynamic:
        if sequential:
            raise ValueError(
                "dynamic requests and problem.loading=at_depot require problem.mode=joint_fleet"
            )
        if preassigned:
            raise ValueError(
                "dynamic requests and problem.loading=at_depot require problem.assignment=fleet"
            )
        if "enabled" not in (config.get("depot_charging") or {}):
            raise ValueError(
                "dynamic request modes must set depot_charging.enabled explicitly: the "
                "depot no longer retires a truck, so its charging role must be stated"
            )
    elif depot.enabled:
        raise ValueError(
            "depot_charging is only defined when the depot does not retire the truck "
            "(requests.mode same_day/drayage or problem.loading=at_depot)"
        )
    if requests.mode == "drayage" and drayage is None:
        raise ValueError("requests.mode=drayage requires a drayage block with terminals")
    if requests.mode != "drayage" and drayage is not None:
        raise ValueError("the drayage block applies only to requests.mode=drayage")
    if loading == "at_depot" and requests.dynamic:
        raise ValueError("problem.loading applies to the static request mode only")
    return ExtensionConfig(
        public_charging=public,
        depot_charging=depot,
        hours_of_service=hos,
        requests=requests,
        drayage=drayage,
        in_flight_information=in_flight,
        loading=loading,
    )
