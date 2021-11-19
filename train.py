"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""


import sys
from collections import OrderedDict

import data
import evaluation.group_evaluator as group_evaluator
import util.util as util
from options.train_options import TrainOptions
from trainers import get_trainer
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)
test_dataloader = data.create_dataloader(util.copyconf(opt, phase="test", batch_size=1))

# create trainer for our model
trainer = get_trainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader.dataset))

# create tool for visualization
visualizer = Visualizer(opt)

# create evaluator for evaluation
evaluator = group_evaluator.GroupEvaluator(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # train generator
        if i % opt.D_steps_per_G == 0:
            for _ in range(opt.num_G_steps):
                trainer.run_generator_one_step(data_i, iter=i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(
                epoch,
                iter_counter.epoch_iter,
                losses,
                iter_counter.time_per_iter,
            )
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict(
                [
                    ('input_label', data_i['label']),
                    ('synthesized_image', trainer.get_latest_generated()),
                    ('real_image', data_i['image']),
                ]
            )
            visualizer.display_current_results(
                visuals, epoch, iter_counter.total_steps_so_far
            )

        if iter_counter.needs_saving():
            print(
                f'saving the latest model (epoch {epoch}, total_steps {iter_counter.total_steps_so_far})'
            )
            trainer.save('latest')
            iter_counter.record_current_iter()

        # Run evaluation
        if iter_counter.needs_evaluation():
            if opt.ema and opt.use_ema and opt.condition_run_stats:
                print('Conditioning netG EMA model...')
                trainer.condition_run_stats(dataloader)

            step = iter_counter.total_steps_so_far
            print(f'Evaluating current model (epoch: {epoch}, total_steps: {step})')
            metrics = evaluator.evaluate(
                train_dataset=dataloader,
                test_dataset=test_dataloader,
                fn_model_forward=trainer.model.generate_visuals_for_evaluation,
            )
            visualizer.plot_current_metrics(metrics, step)
            visualizer.print_current_metrics(metrics, epoch, step)
            for metric_name, metric_val in metrics.items():

                if visualizer.is_best_metric(metric_name, metric_val, step):
                    trainer.save('best_{}'.format(metric_name))

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
        print(
            f'saving the model at the end of epoch {epoch}, iters {iter_counter.total_steps_so_far}'
        )
        trainer.save('latest')
        trainer.save(epoch)


print('Training was successfully finished.')
