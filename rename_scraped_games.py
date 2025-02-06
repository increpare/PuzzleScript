import glob
import re
import shutil

files = glob.glob('data/scraped_games/*')
for file in files:
    new_filename = re.sub("""([?|'|:|+|~|{|}|*|\(|\)"|,|\]|\[])""", "_", file)
    new_filename = re.sub("""(\s)""", "_", new_filename)
    shutil.move(file, new_filename)
    if new_filename != file:
        print(f"Renamed {file} to {new_filename}")
