import os
import re

def main():
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        print("-1")
        return

    files = os.listdir(checkpoint_dir)
    epochs = []
    for f in files:
        match = re.search(r"epoch_(\d+)\.pt", f)
        if match:
            epochs.append(int(match.group(1)))
    
    if not epochs:
        print("-1")
    else:
        print(max(epochs))

if __name__ == "__main__":
    main()
