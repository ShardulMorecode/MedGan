import torch
from generator import Generator

model_path = "D:/Final Year/Mega project/Project/saved_models/generator_v5.pth"  # Update if needed

generator = Generator()
state_dict = torch.load(model_path, map_location=torch.device("cpu"))

print("Model Keys:")
print(state_dict.keys())

print("\nExpected Model Structure:")
for name, param in generator.named_parameters():
    print(name, param.shape)

gen = Generator()
torch.save(gen.state_dict(), "test_model.pth")

gen_test = Generator()
gen_test.load_state_dict(torch.load("test_model.pth", map_location="cpu"))
print("✅ Model saved & loaded successfully!")

