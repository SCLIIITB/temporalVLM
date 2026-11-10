import pandas as pd
import json

# Open the file and load the data

def saveJsonTOCSV(DAP):
    with open(".\\output\\testInput_tomato_growth_stages.json", "r") as file:
        json_data = json.load(file)
    # 2. Parse the nested structure into a flat list of dictionaries
    flattened_rows = []

    for image_name, details in json_data.items():
        # Safely extract the inner growth stage data (handles empty lists safely)
        growth_stage_info = (
            details["growth_stages"][0]
            if details.get("growth_stages")
            else {}
        )

        # Build a clean dictionary for each row
        row = {
            "DAP":DAP,
            "image": image_name,
            "growth_stage": growth_stage_info.get("growth_stage_name"),
            "growth_stage_number": growth_stage_info.get("growth_stage_number"),
            "reason": details.get("reasoning"),
        }
        flattened_rows.append(row)

    # 3. Create the DataFrame
    df = pd.DataFrame(flattened_rows)
    print(df)
    df.to_csv(".\\output\\output1.csv")
    # Print the resulting layout
    print(df)
# saveJsonTOCSV("97")


def addIdAndDapColumns(csv_path, output_path=None):
    # Note looks like "R5T2-a-97DAP": id is everything before the last "-",
    # DAP is the number in front of the literal "DAP" in that last part.
    df = pd.read_csv(csv_path)
    note_parts = df["Note"].str.rsplit("-", n=1, expand=True)
    df["id"] = note_parts[0]
    df["DAP"] = note_parts[1].str.extract(r"(\d+)DAP")

    # id currently looks like "R5T2-a": split further into the plant id ("R5T2")
    # and the photo angle ("a").
    id_angle_parts = df["id"].str.split("-", n=1, expand=True)
    df["id"] = id_angle_parts[0].str.strip()
    df["angle"] = id_angle_parts[1].str.strip().str.strip("-").str.strip()

    df.to_csv(output_path or csv_path, index=False)
    print(df)
    return df
addIdAndDapColumns(".\\output\\labels_dap\\131_notes_times.csv")