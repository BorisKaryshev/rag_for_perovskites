import logging
import logging.config


DEFAULT_LOGGER_CONFIG = {
    'version' : 1,
    "disable_existing_loggers": False,
    'formatters' : {
        'standard' : {
            'format' : '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'minimal' : {
            'format' : '[%(levelname)s]: %(message)s'
        }
    },
    'handlers' : {
        'default' : {
            'level' : 'DEBUG',
            'formatter' : 'minimal',
            'class' : 'logging.StreamHandler',
            'stream' : 'ext://sys.stdout',
        }
    },
    'root': {
        "propagate": True,
        'handlers': ['default'],
        'level': 'DEBUG',
    },
}


def setup_default_logger():
    logging.config.dictConfig(DEFAULT_LOGGER_CONFIG)


def setup_logger(config: dict):
    try:
        if config is None:
            raise ValueError("Config for logger not found. Initializing with default")
        logging.config.dictConfig(config)
    except Exception as ex:
        logging.config.dictConfig(DEFAULT_LOGGER_CONFIG)
        logging.error(f"Error applying logger config: {str(ex)}")
    else:
        logging.info("Applied logger config successfully")
