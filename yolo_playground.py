from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# for i, layer in enumerate(model.)

print(type(model))
print(type(model.model))
print(type(model.model.model))

print(model.model.model[0])

print(model.info())

print(model.model.yaml)