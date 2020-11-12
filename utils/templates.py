# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: templates.py
#  Purpose: Handling  templates used in various plotting scripts
#   Author: Erik Marten Blixt
#   Email: marten.blixt@gmail.com
#
# --------------------------------------------------------------------

def handle_template(template_dict):
    """
    Goes through the given template dictionary and returns values directly useful for the scatter plot
    :param template_dict:
        see x[y,c]templ in calling functions
    :return:
    label, lim, cmap, cnt, bnds, scale
    """
    label = ''
    lim = [None, None]
    cmap = None
    cnt = None
    bnds = None
    scale = 'lin'
    if isinstance(template_dict, dict):
        key_list = list(template_dict.keys())

        if 'full_name' in key_list:
            label += template_dict['full_name']
        if 'unit' in key_list:
            label += ' [{}]'.format(template_dict['unit'])
        if 'min' in key_list:
            try:
                lim[0] = float(template_dict['min'])
            except:
                lim[0] = None
        if 'max' in key_list:
            try:
                lim[1] = float(template_dict['max'])
            except:
                lim[1] = None
        if 'colormap' in key_list:
            cmap = template_dict['colormap']
        if 'center' in key_list:
            cnt = template_dict['center']
        if 'bounds' in key_list:
            bnds = template_dict['bounds']
        if 'scale' in key_list:
            scale = template_dict['scale']

    elif template_dict is None:
        pass
    else:
        raise OSError('Template should be a dictionary')

    return label, lim, cmap, cnt, bnds, scale

