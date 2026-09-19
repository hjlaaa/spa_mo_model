"""Test-only approved NO-OP config projection, never a runtime compatibility layer.

Only the following old default values may be omitted. All other keys, container
order/types and leaves survive; no numeric expected/state/RNG value is rewritten.
"""
NOOP_DEFAULTS = {
    ('graph', 'use_spatial_graph'): True,
    ('graph', 'use_feature_graph'): False,
    ('encoder', 'type'): 'mlp',
    ('encoder', 'residual'): False,
    ('contrastive', 'method'): 'cosie_crossview',
    ('contrastive', 'loss_weight'): 1.0,
    ('contrastive', 'pairwise_all_observed_modalities'): True,
    ('contrastive', 'use_infonce'): False,
    ('contrastive', 'use_temperature'): False,
    ('contrastive', 'use_spot_positive_negative_pairs'): False,
    ('fusion', 'mode'): 'concat_mlp_projection',
    ('fusion', 'input_dim'): 384,
    ('graphsage', 'num_layers'): 1,
    ('graphsage', 'use_distance_weight'): True,
    ('uot', 'initial_from_modalities'): True,
    ('uot', 'use_momentum'): False,
    ('uot', 'momentum'): 0.0,
    ('uot', 'normalize_total_mass'): True,
    ('uot', 'cost'): 'cosine',
    ('reconstruction', 'loss'): 'mse',
    ('loss', 'use_ot_loss'): False,
    ('loss', 'use_spatial_smooth_loss'): False,
    ('loss', 'use_gate_regularization'): False,
}


def project_config(original):
    """Project only explicitly listed old defaults, verifying every other leaf."""
    result=type(original)((key, value) for key,value in original.items())
    for (group,key),expected in NOOP_DEFAULTS.items():
        assert group in original and key in original[group], (group,key)
        actual=original[group][key]
        assert type(actual) is type(expected) and actual==expected, (group,key,actual,expected)
        if result[group] is original[group]:
            result[group]=type(original[group])(original[group])
        del result[group][key]
    def check(before,after,path=()):
        if isinstance(before,dict):
            assert type(after) is type(before)
            assert list(after)==[key for key in before if (*path,key) not in NOOP_DEFAULTS],path
            for key in after:check(before[key],after[key],(*path,key))
        elif isinstance(before,(list,tuple)):
            assert after is before,path
        else:
            assert after is before,path
    check(original,result)
    return result


def project_bundle(original):
    result=type(original)(original)
    result['config']=project_config(original['config'])
    assert list(result)==list(original)
    for key in original:
        if key!='config':assert result[key] is original[key],key
    return result
