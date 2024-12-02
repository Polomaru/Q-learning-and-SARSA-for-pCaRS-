def parse_txt_file(file_path):
    # Inicialización de variables
    name = ""
    type_problem = ""
    comment = ""
    dimension = 0
    cars_number = 0
    edge_weight_type = ""
    edge_weight_format = ""
    edge_weight_matrix = []
    return_rate_matrix = []

    # Abrir el archivo
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Iterar sobre las líneas para extraer la información
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("NAME"):
            name = line.split(":")[1].strip()
        elif line.startswith("TYPE"):
            type_problem = line.split(":")[1].strip()
        elif line.startswith("COMMENT"):
            comment = line.split(":")[1].strip()
        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1].strip())
        elif line.startswith("CARS_NUMBER"):
            cars_number = int(line.split(":")[1].strip())
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":")[1].strip()
        elif line.startswith("EDGE_WEIGHT_FORMAT"):
            edge_weight_format = line.split(":")[1].strip()
        elif line.startswith("EDGE_WEIGHT_SECTION"):
            i += 1  # Salta la línea que indica la sección
            # Leer la matriz EDGE_WEIGHT_SECTION
            temp = []
            while i < len(lines) and not lines[i].startswith("RETURN_RATE_SECTION"):
                row = list(map(int, lines[i].strip().split()))
                if(len(row) == 1): 
                    if len(temp) > 0:
                        edge_weight_matrix.append(temp)
                        temp = []
                else:
                    temp.append(row)
                i += 1
            edge_weight_matrix.append(temp)
            continue  # Para evitar que se incremente i nuevamente
        elif line.startswith("RETURN_RATE_SECTION"):
            i += 1  # Salta la línea que indica la sección
            # Leer la matriz RETURN_RATE_SECTION
            temp = []
            while i < len(lines) and lines[i].strip():
                row = list(map(int, lines[i].strip().split()))
                if(len(row) == 1): 
                    if len(temp) > 0:
                        return_rate_matrix.append(temp)
                        temp = []
                else:
                    temp.append(row)
                i += 1
            return_rate_matrix.append(temp)
            continue  # Para evitar que se incremente i nuevamente

        i += 1

    return dimension, edge_weight_matrix, return_rate_matrix