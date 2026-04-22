from typing import List


def split_into_chunks(file_name: str) -> List[str]:
    with open(file_name, 'r') as file: # open in read-only mode, which is default and can be omitted
        content = file.read()
    return [chunk for chunk in content.split("\n\n")] # chuncking by a blank line instead of breakpoint at end of each line

# testing:
# chunks = split_into_chunks("doc.md")

# for i, chunk in enumerate(chunks):
#     print(f"[{i}] {chunk}\n")