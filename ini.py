"""Contains INI-value handling objects.

This module owns "prod_ini", a dict-type object that represents all product configuration data
obtained from product INI file(s) under the MDW "xyIni" path. It is available for direct
reference through "ini.prod_id" and may be "mocked" (i.e., replaced) during unit
testing.

The "getIni" public function provides access to the value of a specific key under a specific
section. Other public functions are available, but they are intended to be deprecated or are for
debug purposes.

Copyright (C) Seagate Technology LLC, 2021. All rights reserved.
"""
import mdwec
import pynative
import re
import string
import sys
import system.path as path
from   system.type import IniDict


# Precompile the regular expressions for INI parsing
inc_re = re.compile(r'\s*\$i\s+(\w+)(\.[iI][nN][iI])?')
sct_re = re.compile(r'\s*\[([^\]]+)')
key_re = re.compile(r'\s*(\w+)\s*=\s*([^;\r\n]+)')

def _strToBool(str_): return {'true': True, 'false': False}[str_.lower()]

class GetIniParameterError(Exception): pass
class IniError(Exception): pass

def getIni(section, key_name, default = None, esf_call = False):
    """Retrieves INI-file value and handles conversion to native Python types

    The function will pull from the module-level `prod_ini` dict if it is available.
    This is loaded at boot.

    Parameters
    ----------
    section: str
        Section in which desired INI-value is located.
    key_name: str
        Key name labeling desired INI-value.
    default:
        Value to use if/when INI value is not found, for whatever reason.
    esf_call: bool
        True retrieves option value from ESF

    Returns
    -------
        option value as Python native type or string
    """
    sct_key      = (section, key_name)
    have_default = default is not None
    ini_str      = f'[{section}]->{key_name}'

    if not esf_call and 'prod_ini' in globals():

        try:
            val = prod_ini[sct_key]
        except KeyError:
            if have_default:
                print(f'getIni: {ini_str} failed')
                print(f' returning default = {default}')
                return default
            if sct_key not in prod_ini:
                raise GetIniParameterError(f'{ini_str} not found.')
        else:
            return val

    else:

        try:
            val_str = str(pynative.getinistringex(section, key_name)).strip()
        except RcError as e:
            ec = e.rc
            if have_default:
                print(f'getIni: {ini_str} failed')
                print(f' returning default = {default}')
                return default
            if ec == mdwec.XYB_INI_RC_KEYNAME_NOT_FOUND:
                raise GetIniParameterError(f'Keyname "{key_name}" not found.')
            if ec == mdwec.XYB_INI_RC_SECTION_NOT_FOUND:
                raise GetIniParameterError(f'Section "{section}" not found.')
            print(f'Error retrieving {ini_str}')
            raise GetIniParameterError(f'Error retrieving {ini_str}')

        value = mapIniValStr(val_str)

        if value is None:
            if have_default:
                value = default
            else:
                raise GetIniParameterError(f'{ini_str} is empty.')

        return value

def getIniDict(fn_ini, ini=None):
    """Retrieve option values from INI file.

    This function loads an INI configuration file and parses for sections and key values. This
    function is recursive; when an include directive ($i filename) is found, getIniDict is called
    again with that filename, adding on to the building dictionary.

    The cellserv.ini is handled separately because it is found in the persistent flash volume,
    placed there during firmware installation. For any other file, it is downloaded from the host
    and removed afterwards.

    Parameters
    ----------
    fn_ini: str
        File name for file to process.

    Returns
    -------
        Dictionary of section, option, and values.
    """

    # For first time call, create the IniDict, but each recursive call passes this IniDict
    # instance.
    if ini is None:
        ini = IniDict()

    fn_ini = fn_ini.lower()
    if not fn_ini.endswith('.ini'): fn_ini += '.ini'

    if fn_ini == 'cellserv.ini':
        _parseIniFile(path.cell.persistPath('cellserv.ini'), ini, 'cellserv.ini')

    else:
        host_file = path.host.iniPath(fn_ini)
        try:
            with path.host.downloadFile(host_file) as cell_file:
                _parseIniFile(cell_file, ini, fn_ini)
        except path.PathError as err:
            raise IniError(f"Unable to access host file {host_file}: {err}") from None

    return ini

def getIniFileParameter(section, parameter, default = None, trace = None):
    """Retrieve qualified system INI option value.

    Parameters
    ----------
    section: str
        Section in which desired INI-value is located.
    parameter: str
        Keyname labeling desired INI-value.
    default:
        Value to use if/when INI value is not found, for whatever reason.
    trace:
        True displays extra debug information.

     Returns
     -------
        Option value as string
     """
    error = 'No error'
    try:
        value = pynative.getinistringex(section, parameter)
        if not value:
            error = f'Error NoParm: Section [{section}], parameter "{parameter}" ' \
                    f'not defined in prod INI'
            raise GetIniParameterError(error)

        for characters in value:
            if not ((characters in string.digits) or
                    (characters in string.ascii_letters) or
                    (characters in string.punctuation) or
                    (characters == ' ')):
                error = f'Error NotPrintable: Section [{section}], parameter "{parameter}" ' \
                        'contains not printable characters'
                raise GetIniParameterError(error)

    except RcError as err:
        print(f'Exception, {err}')
        _error = ''
        if err.rc == mdwec.XYB_INI_RC_SECTION_NOT_FOUND:
            _error = f'ERROR: NoSection: Section [{section}] is not defined in prod INI'
        elif err.rc == mdwec.XYB_INI_RC_KEYNAME_NOT_FOUND:
            _error = f'ERROR: No parameter: Section [{section}], parameter "{parameter}" ' \
                     'not defined in prod INI'

        if default is None:
            if trace is None:
                print(_error)
            raise GetIniParameterError(error) from None
        value=str(default)
        if trace is None:
            _error += f', use default value: {value}'
            print(_error)
        print(f'Parameter {parameter} use default value: {value}')

    return value

def getSectionKeys(section):
    """ Engineering method to dump a sections keys names.

    Parameters
    ----------
    section: str
        Section in which desired INI-value is located.
     """
    print(key for (sct, key) in prod_ini.keys() if sct == section)

def makeProdIniDict ():
    """ Generate global dictionary of product configuration options. """
    main_system_variables = pynative.getmainsysvars()

    ini_data = getIniDict(main_system_variables.get('FactConfigFile'))

    del main_system_variables['FactMode'] # can change during run-time

    for key, value in main_system_variables.items():
        ini_data['mainsysvars', key] = mapIniValStr(value)

    return ini_data

def mapIniValStr(value_raw):
    """  Map string option values to Python types

    Parameters
    ----------
    value_raw: str
        A string of comma-separated values, all of which must convert to the same type. If the
        string is empty, after stripping whitespace, [None] is the returned value.

    Returns
    -------
        List of converted values, each of the same type of str, int, float, bool.
    """
    #TODO: possible code commonality found with code in
    # wrap.esf_libs.strsToNatives() ?? Ticket?

    values = []
    if value_strip := value_raw.strip():
        if value_strip[0] == '"':
            # Strip double-quote delimiters from string. They are not necessary, but someone may
            # think a string in the INI value should be quoted.
            values = value_strip[1:-1]
        else:
            values = [val.strip() for val in re.split('\s*,\s*', value_strip) if val != '']
            try:
                values = [int(value) for value in values] # hex ini-values not supported
            except ValueError:
                try:
                    values = [int(value, 0) for value in values] # octal ini-values not supported
                except ValueError:
                    try:
                        values = [float(value) for value in values]
                    except ValueError:
                        try:
                            values = [_strToBool(value) for value in values]
                        except KeyError:
                            pass

    n = len(values)

    if   n == 0: return None
    elif n == 1: return values[0]
    else:        return values

def _parseIniFile(cell_file, ini, filename):
    """Read lines from the opened file and process as INI lines, adding to the ini dictionary

    Parameters
    ----------
    cell_file : str
        The path to the file in a cell volume
    ini : IniDict
        The ini dictionary being constructed
    filename : str
        Name of INI file, for error reporting

    Raises
    ------
    IniError:
        Exception for read error, INI syntax error, data conversion error, etc.
    """
    try:
        fhr = open(cell_file)
    except (FileNotFoundError, OSError) as err:
        print(err)
        raise IniError(f"Unable to access cell file {cell_file}: {str(err)}")

    try:
        for line in fhr:
            inc_str = inc_re.match(line)
            if inc_str:
                fn_ini = inc_str.group(1)
                getIniDict(fn_ini, ini)
                continue

            sct_str = sct_re.match(line)
            if sct_str:
                sct = sct_str.group(1)
                continue

            kv_str = key_re.match(line)
            if kv_str:
                key, val_str = kv_str.group(1, 2)

                value = mapIniValStr(val_str)

                if sct is None:
                    raise IniError(f'"{key}" found in "{filename}" without section header')

                ini[sct, key] = value

    except UnboundLocalError:
        raise IniError(f"Key-val pair found in '{filename}' without section header") from None
    except (KeyError, IndexError, TypeError):
        raise IniError(f"INI format error in {filename}") from None
    except OSError as err:
        raise IniError(f"Trouble reading {cell_file}: {err}") from None
    finally:
        fhr.close()


if sys.platform != 'win32': prod_ini = makeProdIniDict()
else                      : prod_ini = {}
