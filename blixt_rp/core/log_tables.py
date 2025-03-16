# Class for handling log tables whos main focus is to relate which Log to use for specific Log types.
#
# It is possible to load several logs under each log type when creating a project. In some  calculations and plots
# it is necessary to choose only one log per log type, and you can achieve that by defining  a *log table* object
#
# It also useful when using cutoffs, as you can specify one cutoff for, say the log type P velocity, and then reuse
# the same cutoffs for different Vp logs by just interchanging the log_table


class LogTable(dict):
    def __init__(self,
                 name: str | None = None,
                 log_table: dict | None = None):
        """

                log_table = {
                   'P velocity': 'vp',
                   'S velocity': 'vs',
                   'Density': 'rhob',
                   'Porosity': 'phie',
                   'Volume': 'vcl'}
        :param name:
        :param log_table:
        """
        self.name = name
        super().__init__(log_table)

    @property
    def invert(self):
        return {v: k for k, v in self.items()}



