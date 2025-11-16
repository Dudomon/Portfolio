import os
import zipfile

# Test paths
current_dir = os.getcwd()
model_path = os.path.join(current_dir, "modelo daytrade", "Legion daytrade.zip")

print(f"Current directory: {current_dir}")
print(f"Model path: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    print("\nZIP contents:")
    with zipfile.ZipFile(model_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            print(f"  - {file}")

# Test enhanced normalizer
norm_path = os.path.join(current_dir, "modelo daytrade", "enhanced_normalizer_final.pkl")
print(f"\nNormalizer path: {norm_path}")
print(f"Normalizer exists: {os.path.exists(norm_path)}")