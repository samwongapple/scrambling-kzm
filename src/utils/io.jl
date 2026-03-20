"""
    save_results(filepath, data::Dict)

Save simulation results to an HDF5 file.
Supports Float64, Vector{Float64}, Matrix{Float64}, Int, String, and Bool values.
"""
function save_results(filepath::String, data::Dict)
    mkpath(dirname(filepath))
    h5open(filepath, "w") do fid
        _write_group(fid, data)
    end
end

function _write_group(group, data::Dict)
    for (key, val) in data
        k = string(key)
        if isa(val, Dict)
            g = create_group(group, k)
            _write_group(g, val)
        elseif isa(val, AbstractArray{<:Number})
            group[k] = collect(Float64, val)
        elseif isa(val, Number)
            group[k] = Float64(val)
        elseif isa(val, AbstractString)
            group[k] = string(val)
        elseif isa(val, Bool)
            group[k] = val ? 1.0 : 0.0
        end
    end
end

"""
    load_results(filepath)

Load simulation results from an HDF5 file into a Dict.
"""
function load_results(filepath::String)::Dict{String,Any}
    data = Dict{String,Any}()
    h5open(filepath, "r") do fid
        _read_group!(data, fid)
    end
    return data
end

function _read_group!(data::Dict, group)
    for key in keys(group)
        obj = group[key]
        if isa(obj, HDF5.Group)
            sub = Dict{String,Any}()
            _read_group!(sub, obj)
            data[key] = sub
        else
            data[key] = read(obj)
        end
    end
end
