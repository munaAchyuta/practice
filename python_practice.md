import os
import json
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import tensorflow as tf
from services.api.core.main import *
from services.api.core import uda
from services.api.utils import proc_data_utils
from services.api.utils.BERT_UDA_Preprocess import BERT_UDA_Preprocess
from services.api.utils.prepare_data import create_test_files

tf.logging.set_verbosity(tf.logging.INFO)

class BERT_UDA_Training(object):
    def __init__(self, do_train, do_eval, do_predict, train_batch_size, eval_batch_size, predict_batch_size,
                 max_seq_length, unsup_data_dir, eval_data_dir, predict_data_dir, vocab_file, bert_config_file,
                 init_checkpoint, model_dir, learning_rate, tsa, num_train_steps=20000, unsup_ratio=3,
                 aug_ops="bt-0.9", aug_copy=1):
        # Enable TF Eager execution
        tfe = tf.contrib.eager
        tfe.enable_eager_execution()

        FLAGS.do_train=do_train
        FLAGS.do_eval=do_eval
        FLAGS.do_predict=do_predict
        FLAGS.train_batch_size = train_batch_size
        FLAGS.eval_batch_size = eval_batch_size
        FLAGS.predict_batch_size = predict_batch_size
        FLAGS.max_seq_length = max_seq_length
        FLAGS.unsup_data_dir=unsup_data_dir
        FLAGS.eval_data_dir=eval_data_dir
        FLAGS.predict_data_dir=predict_data_dir
        FLAGS.bert_config_file=bert_config_file
        FLAGS.vocab_file=vocab_file
        FLAGS.init_checkpoint=init_checkpoint
        FLAGS.model_dir=model_dir
        FLAGS.num_train_steps=num_train_steps
        FLAGS.learning_rate=learning_rate
        FLAGS.unsup_ratio=unsup_ratio
        FLAGS.tsa=tsa
        FLAGS.aug_ops=aug_ops
        FLAGS.aug_copy=aug_copy

    def process(self):
        #ToDO: Bring this label list from processor
        label_list = [0, 1, 2, 3, 4]      
        bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file, FLAGS.model_dropout)
        #tf.gfile.MakeDirs(model_dir)
        '''flags_dict = tf.app.flag_values_dict()
        with tf.gfile.Open(os.path.join(FLAGS.model_dir, "json"), "w") as ouf:
            json.dump(flags_dict, ouf)'''
        save_checkpoints_steps = FLAGS.num_train_steps
        iterations_per_loop = min(save_checkpoints_steps, FLAGS.iterations_per_loop)
        #tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        #        tpu_name, zone=tpu_zone, project=gcp_project)
        tpu_cluster_resolver = None

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
          cluster=tpu_cluster_resolver,
          master=FLAGS.master,
          model_dir=FLAGS.model_dir,
          save_checkpoints_steps=save_checkpoints_steps,
          keep_checkpoint_max=1,
          # train_distribute=train_distribute,
          tpu_config=tf.contrib.tpu.TPUConfig(
              iterations_per_loop=iterations_per_loop,
              per_host_input_for_training=is_per_host))
        model_fn = uda.model_fn_builder(
          bert_config=bert_config,
          init_checkpoint=FLAGS.init_checkpoint,
          learning_rate=FLAGS.learning_rate,
          clip_norm=FLAGS.clip_norm,
          num_train_steps=FLAGS.num_train_steps,
          num_warmup_steps=FLAGS.num_warmup_steps,
          use_tpu=FLAGS.use_tpu,
          use_one_hot_embeddings=FLAGS.use_one_hot_embeddings,
          num_labels=len(label_list),
          unsup_ratio=FLAGS.unsup_ratio,
          uda_coeff=FLAGS.uda_coeff,
          tsa=FLAGS.tsa,
          print_feature=False,
          print_structure=False,
        )
        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        estimator = tf.contrib.tpu.TPUEstimator(
          use_tpu=FLAGS.use_tpu,
          model_fn=model_fn,
          config=run_config,
          params={"model_dir": FLAGS.model_dir},
          train_batch_size=FLAGS.train_batch_size,
          eval_batch_size=FLAGS.eval_batch_size,
          predict_batch_size = FLAGS.predict_batch_size
        )

        if FLAGS.do_train:
            tf.logging.info("  >>> sup data dir : {}".format(FLAGS.sup_data_dir))
            if FLAGS.unsup_ratio > 0:
                tf.logging.info("  >>> unsup data dir : {}".format(FLAGS.unsup_data_dir))

            train_input_fn = proc_data_utils.training_input_fn_builder(
                FLAGS.sup_data_dir,
                FLAGS.unsup_data_dir,
                FLAGS.aug_ops,
                FLAGS.aug_copy,
                FLAGS.unsup_ratio)

        if FLAGS.do_eval:
            tf.logging.info("  >>> dev data dir : {}".format(FLAGS.eval_data_dir))
            eval_input_fn = proc_data_utils.evaluation_input_fn_builder(
                FLAGS.eval_data_dir,
                "clas")

            # ToDO: Bring this label list from processor
            #eval_size = processor.get_dev_size()
            eval_size = 1000
            eval_steps = int(eval_size / FLAGS.eval_batch_size)

        if FLAGS.do_predict:
            tf.logging.info("  >>> predict data dir : {}".format(FLAGS.predict_data_dir))
            predict_input_fn = proc_data_utils.evaluation_input_fn_builder(
                FLAGS.predict_data_dir,
                "clas")

        if FLAGS.do_train and FLAGS.do_eval:
            tf.logging.info("***** Running training & evaluation *****")
            tf.logging.info("  Supervised batch size = %d", FLAGS.train_batch_size)
            tf.logging.info("  Unsupervised batch size = %d",
                            FLAGS.train_batch_size * FLAGS.unsup_ratio)
            tf.logging.info("  Num steps = %d", FLAGS.num_train_steps)
            tf.logging.info("  Base evaluation batch size = %d", FLAGS.eval_batch_size)
            tf.logging.info("  Num steps = %d", eval_steps)
            best_acc = 0
            for _ in range(0, FLAGS.num_train_steps, save_checkpoints_steps):
              tf.logging.info("*** Running training ***")
              estimator.train(
                  input_fn=train_input_fn,
                  steps=save_checkpoints_steps)
              tf.logging.info("*** Running evaluation ***")
              dev_result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
              tf.logging.info(">> Results:")
              for key in dev_result.keys():
                tf.logging.info("  %s = %s", key, str(dev_result[key]))
                dev_result[key] = dev_result[key].item()
              best_acc = max(best_acc, dev_result["eval_classify_accuracy"])
            tf.logging.info("***** Final evaluation result *****")
            tf.logging.info("Best acc: {:.3f}\n\n".format(best_acc))
        elif FLAGS.do_train:
            tf.logging.info("***** Running training *****")
            tf.logging.info("  Supervised batch size = %d", FLAGS.train_batch_size)
            tf.logging.info("  Unsupervised batch size = %d",
                            FLAGS.train_batch_size * FLAGS.unsup_ratio)
            tf.logging.info("  Num steps = %d", FLAGS.num_train_steps)
            estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)
        elif FLAGS.do_eval:
            tf.logging.info("***** Running evaluation *****")
            tf.logging.info("  Base evaluation batch size = %d", FLAGS.eval_batch_size)
            tf.logging.info("  Num steps = %d", eval_steps)
            checkpoint_state = tf.train.get_checkpoint_state(FLAGS.model_dir)

            best_acc = 0
            for ckpt_path in checkpoint_state.all_model_checkpoint_paths:
              if not tf.gfile.Exists(ckpt_path + ".data-00000-of-00001"):
                tf.logging.info(
                    "Warning: checkpoint {:s} does not exist".format(ckpt_path))
                continue
              tf.logging.info("Evaluating {:s}".format(ckpt_path))
              dev_result = estimator.evaluate(
                  input_fn=eval_input_fn,
                  steps=eval_steps,
                  checkpoint_path=ckpt_path,
              )
              tf.logging.info(">> Results:")
              for key in dev_result.keys():
                tf.logging.info("  %s = %s", key, str(dev_result[key]))
                dev_result[key] = dev_result[key].item()
              best_acc = max(best_acc, dev_result["eval_classify_accuracy"])
            tf.logging.info("***** Final evaluation result *****")
            tf.logging.info("Best acc: {:.3f}\n\n".format(best_acc))
        elif FLAGS.do_predict:
            tf.logging.info("***** Running prediction *****")
            print(FLAGS.model_dir)
            checkpoint_state = tf.train.get_checkpoint_state(FLAGS.model_dir)

            best_acc = 0
            for ckpt_path in checkpoint_state.all_model_checkpoint_paths:
                if not tf.gfile.Exists(ckpt_path + ".data-00000-of-00001"):
                    tf.logging.info(
                        "Warning: checkpoint {:s} does not exist".format(ckpt_path))
                    continue
                tf.logging.info("Predicting {:s}".format(ckpt_path))
                
                result = estimator.predict(
                    input_fn=predict_input_fn,
                    checkpoint_path=ckpt_path,
                )
                tf.logging.info(">> Results:")
                preds = [p for p in result]
                #return str(np.argmax(preds, axis=1)[0])
                return np.argmax(preds, axis=1)
    def process_predict(self):
        #ToDO: Bring this label list from processor
        self.label_list = [0, 1, 2, 3, 4]      
        self.bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file, FLAGS.model_dropout)
        #tf.gfile.MakeDirs(model_dir)
        '''flags_dict = tf.app.flag_values_dict()
        with tf.gfile.Open(os.path.join(FLAGS.model_dir, "json"), "w") as ouf:
            json.dump(flags_dict, ouf)'''
        self.save_checkpoints_steps = FLAGS.num_train_steps
        self.iterations_per_loop = min(self.save_checkpoints_steps, FLAGS.iterations_per_loop)
        #tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        #        tpu_name, zone=tpu_zone, project=gcp_project)
        self.tpu_cluster_resolver = None

        self.is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        self.run_config = tf.contrib.tpu.RunConfig(
          cluster=self.tpu_cluster_resolver,
          master=FLAGS.master,
          model_dir=FLAGS.model_dir,
          save_checkpoints_steps=self.save_checkpoints_steps,
          keep_checkpoint_max=1,
          # train_distribute=train_distribute,
          tpu_config=tf.contrib.tpu.TPUConfig(
              iterations_per_loop=self.iterations_per_loop,
              per_host_input_for_training=self.is_per_host))
        self.model_fn = uda.model_fn_builder(
          bert_config=self.bert_config,
          init_checkpoint=FLAGS.init_checkpoint,
          learning_rate=FLAGS.learning_rate,
          clip_norm=FLAGS.clip_norm,
          num_train_steps=FLAGS.num_train_steps,
          num_warmup_steps=FLAGS.num_warmup_steps,
          use_tpu=FLAGS.use_tpu,
          use_one_hot_embeddings=FLAGS.use_one_hot_embeddings,
          num_labels=len(self.label_list),
          unsup_ratio=FLAGS.unsup_ratio,
          uda_coeff=FLAGS.uda_coeff,
          tsa=FLAGS.tsa,
          print_feature=False,
          print_structure=False,
        )
        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        self.estimator = tf.contrib.tpu.TPUEstimator(
          use_tpu=FLAGS.use_tpu,
          model_fn=self.model_fn,
          config=self.run_config,
          params={"model_dir": FLAGS.model_dir},
          train_batch_size=FLAGS.train_batch_size,
          eval_batch_size=FLAGS.eval_batch_size,
          predict_batch_size = FLAGS.predict_batch_size
        )
        
        self.checkpoint_state = tf.train.get_checkpoint_state(FLAGS.model_dir)
        
    def predict(self, data_file_path):
        # for predict CHANGES ======================================================
        #features = BERT_UDA_Preprocess().generate_labeled_features_rtn(data_file_path, '../../../output/predict/', '../../../uncased_L-12_H-768_A-12/vocab.txt')
        BERT_UDA_Preprocess().generate_labeled_features(data_file_path, '../../../output/predict/', '../../../uncased_L-12_H-768_A-12/vocab.txt')
        #proc_data_utils.prediction_input_fn_builder(features, batch_size, feature_specs)
        #features = tf.convert_to_tensor(features, dtype=tf.float32)
        #print("++++++++++++++++++++++++++++++++")
        #print(type(features))
        #print(len(features))
        
        #predict_input_fn = proc_data_utils.prediction_input_fn_builder(features[0])
        predict_input_fn = proc_data_utils.evaluation_input_fn_builder(FLAGS.predict_data_dir, "clas")
        
        best_acc = 0
        for ckpt_path in self.checkpoint_state.all_model_checkpoint_paths:
            if not tf.gfile.Exists(ckpt_path + ".data-00000-of-00001"):
                tf.logging.info(
                    "Warning: checkpoint {:s} does not exist".format(ckpt_path))
                continue
            tf.logging.info("Predicting {:s}".format(ckpt_path))
            result = self.estimator.predict(
                        input_fn=predict_input_fn,
                        checkpoint_path=ckpt_path,
                    )
            tf.logging.info(f">> Results: {result}")
            preds = [p for p in result]
            print("preds : ",preds)
            return str(np.argmax(preds, axis=1)[0])
            #return np.argmax(preds, axis=1)

if __name__ == '__main__':
    #from services.api.utils.BERT_UDA_Preprocess import BERT_UDA_Preprocess
    from pathlib import Path
    import time
    start = time.time()

    # create csv file having two columns . and columns are file name and their class as None.
    '''
    ##text=List of strings to be written to file
    with open('../../../MSA_Docs_csv/csvfile_test.csv','w') as file:
        file.write("filename\tclass_name")
        file.write('\n')
        for ind, filename in enumerate(Path('../../../MSA_Docs/').glob('**/*.txt')):
            line = "{}\t{}".format(filename,None)
            file.write(line)
            file.write('\n')
    print("total files found : {}".format(ind+1))
    '''
    #'''
    for ind in range(0, 36):
        print("=================================")
        print(f"processing file number : {ind}")
        print("=================================")
        path_files_in = f"../../../MSA_Docs_csv/csvfile_test_{ind}.csv"
        path_files_out = f"../../../MSA_Docs_csv/csvfile_out_again_wshuff_{ind}.csv"
        create_test_files(path_files_in)
        BERT_UDA_Preprocess().generate_labeled_features('../../../data/predict/predict.csv', '../../../output/predict/', '../../../uncased_L-12_H-768_A-12/vocab.txt')
        cls = BERT_UDA_Training(False, False, True, 4, 4, 1, 128, '../../../output/', '../../../output/eval/', '../../../output/predict',
                            '../../../uncased_L-12_H-768_A-12/vocab.txt', '../../../uncased_L-12_H-768_A-12/bert_config.json',
                            '../../../uncased_L-12_H-768_A-12/bert_model.ckpt', '../../../model/', 2e-05, 'linear_schedule')
        result = cls.process()
        print("result : ",result[:5])
        result_str = [str(res) for res in result]
        labels = {"0": "ADDENDUM", "1": "MSA", "2": "NDA", "3": "OTHERS", "4": "SOW", "9": "XYZ"}
        pred_classes = [labels[res] for res in result_str]

        with open(path_files_in,'r') as f_in, open(path_files_out, 'w') as f_out:
            # Write header unchanged
            header = f_in.readline()
            f_out.write(header)

            # Transform the rest of the lines
            for ind, line in enumerate(f_in):
                file_name = line.split("\t")[0]
                pred_class = pred_classes[ind]

                f_out.write("{}\t{}".format(file_name,pred_class))
                f_out.write('\n')
    
    #'''
    '''
    # read input csv file line by line , get class from API and save in output csv file.
    # Open both files
    # compare result with csvfile_test_small10.csv
    with open("../../../MSA_Docs_csv/csvfile_test_small10.csv",'r') as f_in, open("../../../MSA_Docs_csv/csvfile_test_small10_againnnn.csv", 'w') as f_out:
        # Write header unchanged
        header = f_in.readline()
        f_out.write(header)

        # Transform the rest of the lines
        cnt = 0
        bug_cnt = 0
        labels = {"0": "ADDENDUM", "1": "MSA", "2": "NDA", "3": "OTHERS", "4": "SOW", "9": "XYZ"}
        
        #=======================
        cls = BERT_UDA_Training(False, False, True, 4, 4, 1, 128, '../../../output/', '../../../output/eval/', '../../../output/predict',
                        '../../../uncased_L-12_H-768_A-12/vocab.txt', '../../../uncased_L-12_H-768_A-12/bert_config.json',
                        '../../../uncased_L-12_H-768_A-12/bert_model.ckpt', '../../../model/', 2e-05, 'linear_schedule')
        cls.process_predict()
        #cls.predict(file_name)
        #=======================
        ''''''
        for ind, line in enumerate(f_in):
            file_name = line.split("\t")[0]
            pred_class = None
            try:
                ''''''
                BERT_UDA_Preprocess().generate_labeled_features(file_name, '../../../output/predict/', '../../../uncased_L-12_H-768_A-12/vocab.txt')
                cls = BERT_UDA_Training(False, False, True, 4, 4, 1, 128, '../../../output/', '../../../output/eval/', '../../../output/predict',
                        '../../../uncased_L-12_H-768_A-12/vocab.txt', '../../../uncased_L-12_H-768_A-12/bert_config.json',
                        '../../../uncased_L-12_H-768_A-12/bert_model.ckpt', '../../../model/', 2e-05, 'linear_schedule')
                result = cls.process()
                ''''''
                result = cls.predict(file_name)
                pred_class = labels[result]
                print("processed file_name : {0}, index : {1}, class: {2}".format(file_name, result, pred_class))
                cnt += 1
            except Exception as e:
                print("Exception :: ",e)
                pred_class = None
                bug_cnt += 1
            f_out.write("{}\t{}".format(file_name,pred_class))
            f_out.write('\n')
        ''''''
        ''''''
        import multiprocessing
        #from itertools import product

        def process_rtr(file_name, pred_class=9):
            result = cls.predict(file_name)
            BERT_UDA_Preprocess().generate_labeled_features(file_name, '../../../output/predict/', '../../../uncased_L-12_H-768_A-12/vocab.txt')
            cls = BERT_UDA_Training(False, False, True, 4, 4, 1, 128, '../../../output/', '../../../output/eval/', '../../../output/predict',
                    '../../../uncased_L-12_H-768_A-12/vocab.txt', '../../../uncased_L-12_H-768_A-12/bert_config.json',
                    '../../../uncased_L-12_H-768_A-12/bert_model.ckpt', '../../../model/', 2e-05, 'linear_schedule')
            result = cls.process()
            labels = {"0": "ADDENDUM", "1": "MSA", "2": "NDA", "3": "OTHERS", "4": "SOW", "9": "XYZ"}
            pred_class = labels[result]
            return '{}\t{}'.format(file_name, pred_class)

        
        #names = ['Brown', 'Wilson', 'Bartlett', 'Rivera', 'Molloy', 'Opie']
        with multiprocessing.Pool(processes=6) as pool:
             results = pool.starmap(process_rtr, [(line.split("\t")[0], 9) for ind, line in enumerate(f_in)])
        print(results)
        ''''''
        ''''''
        import asyncio
        #import random
        
        dict = {'router1': {'id': 1, 'name': 'rtr1_core'},
                'router2': {'id': 2, 'name': 'rt2_core'},
                'router3': {'id': 3, 'name': 'rtr3_access'}}
        
        async def process_rtr(file_name, pred_class=None):
            """Do your execution here."""
            result = cls.predict(file_name)
            pred_class = labels[result]
            #with open(text_dir + post + ".txt" ,"w") as text_file :
            #    text_file.write(text)
            #    text_file.close()
            return pred_class
            #print(f"Processing {post}")

        loop = asyncio.get_event_loop()
        tasks = [asyncio.ensure_future(process_rtr(line.split("\t")[0], None)) for ind, line in enumerate(f_in)]
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()    
        '''

    #print("total files process correctly : {}".format(cnt))
    #print("total files process un-correctly : {}".format(bug_cnt))
    #'''
    
    done = time.time()
    elapsed = done - start
    print("in seconds : ",elapsed)
    print("in minutes : ",elapsed/60)
    # python run_pred_script_log.py > pred_script.log 2>&1 &
