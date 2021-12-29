import matplotlib.pyplot as plt
import os
import blixt_utils.io.io as uio
from blixt_rp.core.well import Project
from blixt_rp.core.minerals import MineralMix
from blixt_rp.core.fluids import FluidMix
from blixt_rp import rp as rp
import blixt_rp.plotting.plot_rp as prp

def main():
    # Create a project
    wp = Project(
        name='FluidSub',
        project_table='excels/project_table.xlsx')

    # Load wells
    wells = wp.load_all_wells(block_name='FBlock')
    w = wells['WELL_F']

    # Load working intervals and templates
    wis = uio.project_working_intervals(wp.project_table)
    templates = uio.project_templates(wp.project_table)

    # Load fluids
    myfluids = FluidMix()
    myfluids.read_excel(wp.project_table)
    print(myfluids.print_all_fluids())

    # Load minerals
    mymins = MineralMix()
    mymins.read_excel(wp.project_table)
    print(mymins.print_all_minerals())

    # Fluid substitution
    cutoffs = {'Volume': ['<', 0.5], 'Porosity': ['>', 0.1]}
    log_table = {'P velocity': 'vp_dry', 'S velocity': 'vs_dry', 'Density': 'rho_dry',
                 'Porosity': 'phie', 'Volume': 'vcl'}

    tag = 'fs'  # tag the resulting logs after fluid substitution
    rp.run_fluid_sub(wells, log_table, mymins, myfluids, cutoffs, wis, tag, templates=templates, block_name='FBlock')

    # plot results
    fig, ax = plt.subplots(figsize=(10, 8))
    log_table = {'P velocity': 'vp_dry', 'S velocity': 'vs_dry', 'Density': 'rho_dry',
                 'Porosity': 'phie', 'Volume': 'vcl'}
    templates['WELL_F']['color'] = 'b'
    prp.plot_rp(wells, log_table, wis, 'SAND E', cutoffs, templates, fig=fig, ax=ax, edge_color=False, show_masked=True,
                block_name='FBlock')

    # specify the new logs coming from the fluid substitution
    log_table = {'P velocity': 'vp_dry_fs', 'S velocity': 'vs_dry_fs', 'Density': 'rho_dry_fs',
                 'Porosity': 'phie', 'Volume': 'vcl'}
    templates['WELL_F']['color'] = 'r'
    prp.plot_rp(wells, log_table, wis, 'SAND E', cutoffs, templates, fig=fig, ax=ax, edge_color=False,
                block_name='FBlock')

    # specify the new logs coming from the RokDoc fluid substitution stored in the las file
    log_table = {'P velocity': 'vp_so08', 'S velocity': 'vs_so08', 'Density': 'rho_so08',
                 'Porosity': 'phie', 'Volume': 'vcl'}
    templates['WELL_F']['color'] = 'g'
    prp.plot_rp(wells, log_table, wis, 'SAND E', cutoffs, templates, fig=fig, ax=ax, edge_color=False,
                block_name='FBlock')

    del(wells['WELL_F'].block['FBlock'].logs['rho_so08'])
    del(wells['WELL_F'].block['FBlock'].logs['rho_sg08'])
    wells['WELL_F'].depth_plot('Density')


    # Save results
    for well in wells.values():
        uio.write_las(
            os.path.join(wp.working_dir, 'results_folder', '{}.las'.format(well.well)),
            well.header,
            well.block['FBlock'].header,
            well.block['FBlock'].logs
        )

    plt.show()

if __name__ == '__main__':
    main()