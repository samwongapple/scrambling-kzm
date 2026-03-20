"""
    load_config(config_path; defaults_path=nothing)

Load a TOML config file, optionally merging with a defaults file.
The config file values override defaults.
"""
function load_config(config_path::String; defaults_path::Union{String,Nothing}=nothing)::Dict{String,Any}
    config = TOML.parsefile(config_path)

    if defaults_path !== nothing && isfile(defaults_path)
        defaults = TOML.parsefile(defaults_path)
        config = _merge_dicts(defaults, config)
    end

    return config
end

"""
Recursively merge two dictionaries. Values in `override` take precedence.
"""
function _merge_dicts(base::Dict, override::Dict)::Dict{String,Any}
    result = Dict{String,Any}()
    for (k, v) in base
        result[k] = v
    end
    for (k, v) in override
        if haskey(result, k) && isa(result[k], Dict) && isa(v, Dict)
            result[k] = _merge_dicts(result[k], v)
        else
            result[k] = v
        end
    end
    return result
end
