def minimum_working_example():
    from bokeh.models import ColumnDataSource, DataTable, TableColumn, CheckboxEditor, CustomJS
    from bokeh.io import output_file

    output_file('C:\\Users\\emb\\Downloads\\table.html')


    # Main table
    car_source = ColumnDataSource(dict(
        car_brand=['Honda', 'Toyota'],
        reliable=[True, True]
    ))

    table_columns = [
        TableColumn(field='car_brand', title='Car manufacturer'),
        TableColumn(field='reliable', title='Reliable?', editor=CheckboxEditor())
    ]
    main_table = DataTable(
        source=car_source,
        columns=table_columns,
        editable=True,
        width= 200
    )

    # Secondary CDS
    rental_source = ColumnDataSource(dict(
        car_rental=['Sixt', 'Seventh', 'Eighth'],
        car_brand=['Honda', 'Honda', 'Toyota'],
        in_business=[True, True, True]
    ))

    # callback functions
    callback1 = CustomJS(
        args=dict(car_source=car_source, rental_source=rental_source),
        code = """
            const car_data = car_source.data;
            var rental_data = rental_source.data;
            for (let i = 0; i < car_data['car_brand'].length; i++) {
                let this_car_brand = car_data['car_brand'][i];
                for (let j = 0; j < rental_data['car_rental'].length; j++) {
                    if (rental_data['car_brand'][j] === this_car_brand) {
                        rental_data['in_business'][j] = car_data['reliable'][i];
                        console.log('Car rental ' + rental_data['car_rental'][j] + ' in business? ' + rental_data['in_business'][j]);
                    }
                } 
            }
            rental_source.data = rental_data;
            rental_source.change.emit();
        """
    )

    callback2 = CustomJS(code="""
        console.log('Car rentals has been updated');
    """)

    def python_callback1(attr, old, new):
        print('Cars have been updated in Python' )

    def python_callback2(attr, old, new):
        print('Car rentals has been updated in Python' )

    car_source.js_on_change('patching', callback1)
    car_source.on_change('data', python_callback1)

    rental_source.js_on_change('data', callback2)
    rental_source.on_change('data', python_callback2) # This does not work either!

    return main_table


if __name__ == '__main__':
    from bokeh.plotting import show
    table = minimum_working_example()
    show(table)