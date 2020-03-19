import re
import json

class LogWrapper:
    def __init__(self):
        self.__log = {"version":{},"well":{},"curve":{},"parameter":{},"data" :{},"other":""}
    
    @property
    def data(self): return self.__log["data"]
    @property
    def version(self): return self.version_section["vers"]["value"]
    @property
    def version_section(self): return self.__log["version"]
    @property
    def well(self): return self.__log["well"]
    @property
    def curve(self): return self.__log["curve"]
    @property
    def parameter(self): return self.__log["parameter"]
    @property
    def other(self): return self.__log["other"]
    
    def add(self,section,mnem,content):
        if section == "data":
            if isinstance(content,list):
                self.__log[section][mnem] = content
            else:
                self.__log[section][mnem].append(content)
        elif section=="other":
            self.__log[section] = self.other + "".join(str(content).strip())
        else:
            self.__log[section][mnem] = content

    def get_json(self,out="outfile"):
        """Write out.json to disk and Return json format."""
        out = out +".json"
        with open(out, "w") as f :
            json.dump(self.__log, f, indent=4)
        return json.dumps(self.__log, indent=4)
    
    def get_dict(self):
        """Return dict({"version":{},"well":{},"curve":{},"parameter":{},"data" :{},"other":""})"""
        return self.__log
    


class Converter():
    
    def __init__(self):
        self.supported_version = {2.0, 3.0}
        self.__generated_keys = []
        self.__null_val = None
        self.__lines = None
    
    def __str__(self):
        return "Supported LAS Version : {0}".format(self.supported_version)

    def set_file(self, file):
        """Convert file and Return `self`. """
        file_format = None
        ext = file.rsplit(".", 1)[-1].lower()
        
        if ext == "las":            
            file_format = 'las'
        
        elif ext == 'txt': # text files from Avseth & Lehocki in the  Spirit study 
            file_format = 'RP well table'
        
        else:            
            raise Exception("File format '{}'. not supported!".format(ext))
        
        with open(file, "r") as f:
            self.__lines = f.readlines()
        return self.__convert(file_format=file_format)  # read all lines from data

    def set_stream(self,stream):
        """Convert file and Return `self`. """
        self.__lines = stream
        self.__convert()
        return self.__convert()  # read all lines from data

    def __parse(self, x):
        try:
            x = int(x)
        except ValueError:
            try:
                x = float(x)
            except ValueError:
                pass
        return x
    
    def __convert(self, file_format='las'):
        section = ""
        rules = {"version", "well", "parameter", "curve"}
        descriptions = []
        curve_names = None
        log = LogWrapper()
        if file_format == 'RP well table':
            self.__null_val = 'NaN'

        for line in self.__lines:
            content = {}

            if isinstance(line, bytes):
                line = line.decode("utf-8").strip()

            # line just enter or "\n"
            if len(line) <= 1 : continue
            # comment
            if "#" in line: continue

            # section
            if "~" in line:
                section = self.__get_current_section(line)
                
                # get section version first
                if section == "version": continue

                if log.version not in self.supported_version:
                    raise Exception("Version not supported!")

                # generate keys of log[data] based on log[curve]
                if section == "data":                    
                    self.__generated_keys = [e.lower() for e in log.curve.keys()]
                    for key in self.__generated_keys:
                        #inital all key to empty list
                        log.add(section,key,[])

                continue
            
            if file_format == 'RP well table':
                if line[:7] == 'Columns':
                    section = 'RP header'
                    continue  # jump into header

                if section == 'RP header':
                    if line[2:5] == ' - ':
                        descriptions.append(line.split(' - ')[-1].strip())

                if line[:7] == 'Well ID':
                    # parse curve names
                    curve_names = [t.strip().lower() for t in line.split('\t')]
                    section = 'dummy_value'

                if line[:4] == '  No':
                    # parse line of units
                    #unit_names = [t.strip() for t in line.split('\t')]
					unit_names = [t.strip() for t in line.split()]
                    unit_names = [t.replace('[', '').replace(']', '') for t in unit_names]

                    for this_curve_name, this_unit_name , this_description in zip(curve_names, unit_names, descriptions):
                        log.add('curve',
                                this_curve_name,
                                {'api_code': None, 'unit': this_unit_name, 'desc': this_description}
                        )
                    self.__generated_keys = [key for key in curve_names]
                    section = 'data'
                    # initiate all key to empty list
                    for key in self.__generated_keys:
                        log.add(section,key,[])
                    continue  # jump into data

            # unregistered section
            if section is None: continue

            if section in rules:

                # index of seperator
                if re.search("[.]{1}", line) is None:
                    print('Caught problem')
                    continue
                mnem_end = re.search("[.]{1}", line).end()
                unit_end = mnem_end + re.search("[ ]{1}", line[mnem_end:]).end()
                colon_end = unit_end + re.search("[:]{1}", line[unit_end:]).start()

                # divide line
                mnem = line[:mnem_end-1].strip()
                unit = line[mnem_end:unit_end].strip()
                data = line[unit_end:colon_end].strip()
                desc = line[colon_end+1:].strip()

                # convert empty string ("") to None
                if len(data) == 0: data = None
                if section == "well" and mnem == "NULL":
                    # save standard LAS NULL value
                    self.__null_val = data
                    data = None

                # parse data to type bool or number
                # BUT it also parsed well names as floats, which we should avoid
                if data is not None:
                    if desc == 'WELL':
                        # avoid the __parse() function
                        pass
                    elif data == "NO":
                        data = False
                    elif data == "YES":
                        data = True
                    else:
                        data = self.__parse(data)
                
                # dynamic key
                key = "api_code" if section == "curve" else "value"
                content = {
                    key: data,
                    "unit": unit,
                    "desc": desc
                }

                log.add(section, mnem.lower(), content)

            elif section == "data":
                content = line.split()
                for k, v in zip(self.__generated_keys, content):
                    v = float(v) if v != self.__null_val else None
                    log.add(section, k.lower(), v)

            elif section == "other":
                log.add(section,None,line)

        return log

    def __get_current_section(self,line):
        if '~V' in line : return 'version'
        if '~W' in line: return 'well'
        if '~C' in line: return 'curve'
        if '~P' in line: return 'parameter'
        if '~O' in line: return 'other'
        if '~A' in line: return 'data'
        # ~ Unregistered section
        return None
