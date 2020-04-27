# FYP_Backend

Repository for FYP Backend

[datatype] = "gigaword", "reuters", "cnn"

# Install Packages (requires Python3 installation)

pip install -r requirements.txt

# Data

main.py --action=data --datatype=[datatype]

# Pretrain

main.py --action=pretrain --datatype=[datatype]

# Train

main.py --action=train --datatype=[datatype]
main.py --action=train --datatype=[datatype] --load

# Test

main.py --action=test --datatype=[datatype]

# Save Model

main.py --action=save

# Run Server

server.py

# Datasets (extract contents in data folder)

https://drive.google.com/file/d/1CgDz3z-ORGWRiyFldHcR7x2szvWJN1xC/view?usp=sharing

# Trained Models

https://drive.google.com/file/d/1UFo8ZgzmIC9HOkUA15lOguiqq56OqiqI/view?usp=sharing