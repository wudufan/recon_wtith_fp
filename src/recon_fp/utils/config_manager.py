'''
Support for modification of configuration content through argparser.
'''

import ast
import configparser


# %%
def parse_config_with_extra_arguments(parser, cmds=None):
    '''
    Return:
    @args: the argparser parsed results
    @config: updated configparser class
    @train_args: type-converted dictionary of the training arguments
    '''

    args, _ = parser.parse_known_args(cmds)
    config = configparser.ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.read(args.config)

    # then modify the configuration with any additional arguments
    parser = build_argparser_from_config(parser, config)
    args = parser.parse_args(cmds)
    config = update_config_from_args(config, args)

    train_args = get_kwargs(config)

    return args, config, train_args


def build_argparser_from_config(parser, config):
    for sec in config:
        for k in config[sec]:
            arg_name = sec + '.' + k
            default_val = config[sec][k]

            parser.add_argument('--' + arg_name, default=default_val)

    return parser


def update_config_from_args(config, args):
    for k in vars(args):
        tokens = k.split('.')
        if len(tokens) == 2:
            config[tokens[0]][tokens[1]] = getattr(args, k)

    return config


def get_kwargs(config, verbose=1):
    arg_dict = {}
    for sec in config:
        # get default type for the section
        arg_dict[sec] = {}
        for k in config[sec]:
            # determine the type of the property
            try:
                if len(config[sec][k]) == 0:
                    arg_dict[sec][k] = None
                else:
                    arg_dict[sec][k] = ast.literal_eval(config[sec][k])
            except Exception:
                if verbose > 0:
                    print('unparsed config at [{0}] {1} = {2}'.format(sec, k, config[sec][k]))
                arg_dict[sec][k] = config[sec][k]

    return arg_dict
