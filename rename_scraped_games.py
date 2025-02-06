import glob
import re
import shutil

files = glob.glob('data/scraped_games/*')
for file in files:
    new_filename = re.sub("""([?|'|:|*|\(|\)"|,])""", "_", file)
    shutil.move(file, new_filename)
    print(f"Renamed {file} to {new_filename}")
