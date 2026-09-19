"""P0-only projection of six P2-retired input fields now rejected by B8.

These keys were removed from production defaults in P2; frozen P0 inputs kept
them. They are independent of B8's 23 new NO-OP default removals.
"""
P2_RETIRED_DEFAULTS = {
    ('uot', 'tol'): 1e-6,
    ('uot', 'check_every'): 10,
    ('uot', 'clip_cost_min'): 0.0,
    ('uot', 'clip_cost_max'): 2.0,
    ('uot', 'keep_dense'): False,
    ('ot_attention', 'direction'): 'forward',
}


def project_p2_config_bundle(original):
    config=original['config']
    projected=type(config)(config)
    for (group,key),expected in P2_RETIRED_DEFAULTS.items():
        actual=config[group][key]
        assert type(actual) is type(expected) and actual==expected,(group,key,actual,expected)
        if projected[group] is config[group]:projected[group]=type(config[group])(config[group])
        del projected[group][key]
    def identity(before,after,path=()):
        if isinstance(before,dict):
            assert type(after) is type(before)
            assert list(after)==[key for key in before if (*path,key) not in P2_RETIRED_DEFAULTS],path
            for key in after:identity(before[key],after[key],(*path,key))
        else:assert after is before,path
    identity(config,projected)
    result=type(original)(original)
    result['config']=projected
    assert list(result)==list(original)
    for key in original:
        if key!='config':assert result[key] is original[key],key
    return result
