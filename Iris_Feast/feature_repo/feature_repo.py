
from datetime import timedelta
from feast import BigQuerySource, FeatureView, FeatureService, Entity, ValueType

# Define flower species as entity
flower_entity = Entity(
    name="species",
    description="A Species of Iris Flower",
    value_type=ValueType.STRING
)

# Define feature view for flower measurements
flower_features = FeatureView(
    name="flower_features",
    entities=[flower_entity],
    ttl=timedelta(weeks=52),  # Time-to-live for features
    source=BigQuerySource(
        table=f"arcane-rigging-461217-m1.Iris_Dataset.Iris_Table",
        timestamp_field="event_timestamp"
    ),
    tags={"assignment":"week_3"}
)

# Create feature service for one model version
# FeatureService groups features for specific use cases
model_v1 = FeatureService(
    name="feast_model_v1",
    features=[flower_features]
)
