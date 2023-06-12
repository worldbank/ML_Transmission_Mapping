from configparser import ConfigParser

from psycopg2.extras import NamedTupleCursor


def get_db_config(filename=None, section="postgresql"):
    if filename is None:
        filename = f"{__file__}/../database.ini"
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception("Section {0} not found in the {1} file".format(section, filename))
    db["cursor_factory"] = NamedTupleCursor

    return db
