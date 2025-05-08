import os 

def count_in_file(file_path):
    # init counters
    counters = {
        'case_nums': 0,
        'persons': 0,
        'institutions': 0,
        'prom_dates': 0,
        'republic_acts': 0,
        'statutes': 0,
        'constitutes': 0,
    }

    # map from entity class to counter key
    entity_map = {
        'CASE_NUM': 'case_nums',
        'PERSON': 'persons',
        'INS': 'institutions',
        'PROM_DATE': 'prom_dates',
        'RA': 'republic_acts',
        'STA': 'statutes',
        'CNS': 'constitutes',
    }

    # check if currently inside an entity
    inside_entity = {key: False for key in entity_map.keys()}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            token, tag = parts

            for entity_type in entity_map:
                if tag == f'B-{entity_type}':
                    counters[entity_map[entity_type]] += 1
                    inside_entity[entity_type] = True
                elif tag != f'I-{entity_type}':
                    inside_entity[entity_type] = False

    return counters

def tally_folder(folder_path):
    file_total = {}
    combined_totals = {
        'case_nums': 0,
        'persons': 0,
        'institutions': 0,
        'prom_dates': 0,
        'republic_acts': 0,
        'statutes': 0,
        'constitutes': 0,
    }

    for filename in os.listdir(folder_path):
        if filename.endswith('.iob'): 
            file_path = os.path.join(folder_path, filename)
            counts = count_in_file(file_path)
            file_total[filename] = counts

        # get totals
        for key in combined_totals:
            combined_totals[key] += counts.get(key, 0)

    return file_total, combined_totals

# main
if __name__ == "__main__":

    while True:
        folder = input("Folder name: ").strip()

        if not os.path.isdir(folder):
            print("Folder not found or invalid input. Please try again.\n")
            continue

        print("Folder exists! Checking now...\n")
        result, summary = tally_folder(folder)

        # print results per file
        for file, counts in result.items():
            print(f"\n{file}:")
            for entity, count in counts.items():
                print(f"  {entity}: {count}")

        # print entire summary
        print("\n\n")
        print(f"\"{folder}\" Summary:")
        for entity, count in summary.items():
            print(f" {entity}: {count}")  