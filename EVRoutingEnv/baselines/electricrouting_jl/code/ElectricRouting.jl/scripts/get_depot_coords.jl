# script to print depot (first node) coordinates for each instance file
using Printf

dir = length(ARGS) >= 1 ? ARGS[1] :
    "data/evrptw/Instances"

files = filter(name -> isfile(joinpath(dir, name)), readdir(dir))

if isempty(files)
    @printf("No files found in %s\n", dir)
    exit()
end

# regex for common "index x y" coordinate lines (TSPLIB-like)
coord_line_re = r"^\s*\d+\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)"

for name in sort(files)
    path = joinpath(dir, name)
    txt = try
        read(path, String)
    catch err
        @printf("%s: error reading file: %s\n", name, err)
        continue
    end

    lines = split(txt, '\n')

    # 1) Try to find a NODE_COORD_SECTION style block first (TSPLIB-like)
    idx = findfirst(l -> occursin(r"(?i)NODE_COORD_SECTION", l) || occursin(r"(?i)NODE_COORD", l), lines)
    start_line = idx === nothing ? 1 : idx + 1

    found = false
    for i in start_line:length(lines)
        line = strip(lines[i])
        # stop if reached a different section or end marker
        if isempty(line) || startswith(uppercase(line), "EOF") || occursin(r"(?i)DEMAND_SECTION|DEPOT_SECTION|VEHICLE|SERVICE_TIME", line)
            continue
        end
        m = match(coord_line_re, line)
        if m !== nothing
            x = parse(Float64, m.captures[1])
            y = parse(Float64, m.captures[2])
            @printf("%s\t%.6f\t%.6f\n", name, x, y)
            found = true
            break
        end
    end

    # 2) If not found, handle the plain table format where the first non-header line
    #    is the depot (e.g. header line followed by "D0 d 40 50 ..." — x and y are tokens 3 and 4)
    if !found
        first_non = findfirst(l -> !isempty(strip(l)), lines)
        if first_non !== nothing
            # find the next non-empty line after the header (often the depot)
            j = first_non + 1
            while j <= length(lines) && isempty(strip(lines[j]))
                j += 1
            end
            if j <= length(lines)
                tokens = split(strip(lines[j]))
                if length(tokens) >= 4
                    ok = true
                    try
                        x = parse(Float64, tokens[3])
                        y = parse(Float64, tokens[4])
                    catch
                        ok = false
                    end
                    if ok
                        @printf("%s\t%.6f\t%.6f\n", name, x, y)
                        found = true
                    end
                end
            end
        end
    end

    # 3) Final fallback: search whole file for first "index x y" occurrence
    if !found
        m_all = match(coord_line_re, txt)
        if m_all !== nothing
            x = parse(Float64, m_all.captures[1])
            y = parse(Float64, m_all.captures[2])
            @printf("%20s\t%.6f\t%.6f\n", name, x, y)
        else
            @printf("%20s\tDepot coordinates not found\n", name)
        end
    end
end
