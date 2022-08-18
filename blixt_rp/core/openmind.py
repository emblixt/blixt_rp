import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties


# To test blixt_rp and blixt_utils libraries directly, without installation:
sys.path.append('C:\\Users\\eribli\\PycharmProjects\\blixt_rp')
sys.path.append('C:\\Users\\eribli\\PycharmProjects\\blixt_utils')

import openmind.openmind_tool as openmind_tool
import openmind.core.models as models


def get_info():
    with openmind_tool.OpenMind_Tool() as omt:
        surveys = omt.surveys
        version = omt.version

        for survey in surveys:
            print('Survey: {}'.format(survey.name))
            for cube in survey.seismic_cubes:
                print('  Cube: {}'.format(cube.name))
                metadata_reply = omt.get_metadata(cube.id)
                print('  Cube metadata: {}'.format(metadata_reply))
            for voi in survey.volumes_of_interests:
                print('  VOI: {}'.format(voi.name))


def find_first_survey_by_name(survey_name: str, omt: openmind_tool.OpenMind_Tool):
    """
    Finds the (first) survey id for the given survey_name in the open OpenMind project.

    survey_name:
        string
    omt:
        openmind_tool.OpenMind_Tool

    :return
        string
        cube id
    """
    for survey in omt.surveys:
        if survey.name == survey_name:
            return survey.id
    return ''


def find_first_cube_by_name(cube_name: str, omt: openmind_tool.OpenMind_Tool):
    """
    Finds the (first) cube id for the given cube_name in the open OpenMind project.

    cube_name:
        string
    omt:
        openmind_tool.OpenMind_Tool

    :return
        string
        cube id
    """
    for survey in omt.surveys:
        for cube in survey.seismic_cubes:
            if cube.name == cube_name:
                return cube.id
    return ''


def get_trace(omt: openmind_tool.OpenMind_Tool,
               _cube_name: str,
               i_in_percent: float,
               j_in_percent: float):
    for survey in omt.surveys:
        for cube in survey.seismic_cubes:
            if cube.name == _cube_name:
                metadata = cube.metadata
                i = int( i_in_percent * metadata.size.i / 100.)
                j = int( j_in_percent * metadata.size.j / 100.)
                return cube.get_trace(i, j, 1)

