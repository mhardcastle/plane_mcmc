option_list = ( ( 'regions', 'jet', str, None, 'Region file for the jet'),
                ( 'regions', 'counterjet', str, None, 'Region file for the counterjet'),
                ( 'source', 'z', float, None, 'Redshift'),
                ( 'source', 'name', str, 'source', 'Source name, used to name the save file'),
                ( 'priors', 'incangle', list, [0,90], 'Inclination angle prior (degrees)'),
                ( 'priors', 'openangle', list, [0,45], 'Opening angle prior (degrees)'),
                ( 'priors', 'posangle', list, [0,360], 'Position angle prior (degrees)'),
                ( 'priors', 'period', list, [-1,1], 'Precession period prior (log10 t/Myr)'),
                ( 'priors', 'speed', list, [0.1,0.9999], 'Jet speed prior (fraction of c)') )
                 
