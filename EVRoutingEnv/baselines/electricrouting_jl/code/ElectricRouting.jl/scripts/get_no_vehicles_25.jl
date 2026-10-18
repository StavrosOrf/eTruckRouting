# quick script to extract NoVehicles from files whose names end with "_25"
using Printf

dir = length(ARGS) >= 1 ? ARGS[1] :
    "data/evrptw/Solutions/MP"

files = filter(
    name -> (
        isfile(joinpath(dir, name)) 
        && occursin(r"_100$", name)
    ), 
    readdir(dir)
)

# if isempty(files)
#     @printf("No files ending in \"_25\" found in %s\n", dir)
#     exit()
# end

for name in sort(files)
    path = joinpath(dir, name)
    txt = try
        read(path, String)
    catch err
        @printf("%s: error reading file: %s\n", name, err)
        continue
    end
    m = match(r"NoVehicles\s*=\s*(\d+)\s*;?", txt)
    if m !== nothing
        @printf("%s\t%d\n", name, parse(Int, m.captures[1]))
    else
        @printf("%s\tNoVehicles not found\n", name)
    end
end
