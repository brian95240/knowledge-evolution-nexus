logging.config.dictConfig({
       "version": 1,
       "disable_existing_loggers": False,
       "formatters": {
           "json": {
               "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
               "fmt": "%(timestamp)s %(level)s %(name)s %(message)s"
           }
       },
       "handlers": {
           "json": {
               "class": "logging.StreamHandler",
               "formatter": "json"
           }
       },
       "root": {
           "handlers": ["json"],
           "level": "INFO"
       }
   })