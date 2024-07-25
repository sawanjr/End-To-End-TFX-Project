import os

import tensorflow_model_analysis as tfma
from tfx import v1 as tfx
from tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2


TRAIN_STEPS = 50000
EVAL_STEPS = 10000


def init_components(
    data_dir, # Path where the training/eval data can be found. 
    module_file, #Python module required by the Transform and Trainer components
    training_steps=TRAIN_STEPS,
    eval_steps=EVAL_STEPS,
    serving_model_dir=None, #Path where the exported model should be stored
    vertex_training_custom_config=None,
    vertex_serving_args=None,
):

    if serving_model_dir and vertex_serving_args:
        raise NotImplementedError(
            "Can't set vertex_serving_args and serving_model_dir at "
            "the same time. Choose one deployment option."
        )

    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=9),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=1),
            ]
        )
    )

    example_gen = tfx.components.CsvExampleGen(
        input_base=os.path.join(os.getcwd(), data_dir), output_config=output
    )

    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs["examples"]
    )

    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=False,
    )

    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    transform = tfx.components.Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=module_file,
    )

    training_kwargs = {
        "module_file": module_file,
        "examples": transform.outputs["transformed_examples"],
        "schema": schema_gen.outputs["schema"],
        "transform_graph": transform.outputs["transform_graph"],
        "train_args": trainer_pb2.TrainArgs(num_steps=training_steps),
        "eval_args": trainer_pb2.EvalArgs(num_steps=eval_steps),
    }

    if vertex_training_custom_config:
        training_kwargs.update({"custom_config": vertex_training_custom_config})
        trainer = tfx.extensions.google_cloud_ai_platform.Trainer(**training_kwargs)
    else:
        trainer = tfx.components.Trainer(**training_kwargs)

    model_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.dsl.Channel(type=tfx.types.standard_artifacts.ModelBlessing),
    )

    eval_config_threshold = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(signature_name="serving_default",preprocessing_function_names=['transform_features'],label_key='ocean_proximity_xf')],
    slicing_specs=[
        tfma.SlicingSpec(),
        tfma.SlicingSpec(feature_keys=['housing_median_age'])
    ],
    metrics_specs=[
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name='AUC'),
            tfma.MetricConfig(class_name='ExampleCount'),
            tfma.MetricConfig(
                class_name='CategoricalAccuracy',
                threshold=tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(lower_bound={'value': 0.44}),
                    change_threshold=tfma.GenericChangeThreshold(
                        direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                        absolute={'value': 0.01}
                    )))])])


    evaluator = tfx.components.Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )

    if vertex_serving_args:
        pusher = tfx.extensions.google_cloud_ai_platform.Pusher(
            model=trainer.outputs["model"],
            model_blessing=evaluator.outputs["blessing"],
            custom_config=vertex_serving_args,
        )

    elif serving_model_dir:
        pusher = tfx.components.Pusher(
            model=trainer.outputs["model"],
            model_blessing=evaluator.outputs["blessing"],
            push_destination=pusher_pb2.PushDestination(
                filesystem=pusher_pb2.PushDestination.Filesystem(
                    base_directory=serving_model_dir
                )
            ),
        )
    else:
        raise NotImplementedError(
            "Provide ai_platform_serving_args or serving_model_dir."
        )

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    ]
    return components
