import tensorflow as tf
import numpy as np
from data import data
from utilities import *
import os
from rouge import Rouge
import json

RECONSTRUCTOR_BASE = 0.0
RECONSTRUCTOR_WEIGHT = 0.5
RECONSTRUCTOR_ITERATIONS = 3
DISCRIMINATOR_ITERATIONS = 3
PRETRAIN_DISCRIMINATOR_RECONSTRUCTOR_STEPS = 500
COVERAGE_WEIGHT = 0.1
LMBDA = 10
ARTICLE_LENGTH = 50
SUMMARY_LENGTH = 14
BATCH_SIZE = 64
PRETRAIN_NUM_STEPS = 20000
TRAIN_NUM_STEPS = 7000
PRETRAIN_CHECKPOINT = 1000
TRAIN_CHECKPOINT = 100
EMBEDDING_DIMENSION = 300

class model():

    def __init__(self, sess, args):

        self.sess = sess
        self.action = args.action
        self.datatype = args.datatype
        self.load = args.load
        self.data = data(args)
        self.vocab_size = self.data.vocab_size
        self.summary_lstm_length = [SUMMARY_LENGTH + 1 for _ in range(BATCH_SIZE)]

        self.build_layers()

        for v in tf.trainable_variables():
            print(v.name,v.get_shape().as_list())
        
        self.generator_saver = tf.train.Saver(self.generator_variables, max_to_keep=10)
        if self.action != 'test':
            self.discriminator_saver = tf.train.Saver(self.discriminator_variables, max_to_keep=2)
            self.reconstructor_saver = tf.train.Saver(self.reconstructor_variables, max_to_keep=2)

    def build_layers(self):

        with tf.variable_scope('input_layer') as scope:

            begin_of_sequence = tf.ones([BATCH_SIZE, 1], dtype=tf.int32) * 1
            end_of_sequence = tf.ones([BATCH_SIZE, 1], dtype=tf.int32) * 0

            self.source = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE, ARTICLE_LENGTH))
            self.real_samples = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE, SUMMARY_LENGTH))

            self.decoder_reconstructor_inputs = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE, ARTICLE_LENGTH))
            decoder_reconstructor_inputs = tf.concat([begin_of_sequence, self.decoder_reconstructor_inputs], axis = 1)

            decoder_reconstructor_targets = tf.concat([self.source, end_of_sequence], axis=1)

            generator_decoder_inputs = tf.ones([BATCH_SIZE, SUMMARY_LENGTH + 1], dtype=tf.int32)

            self.generator_target = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE, SUMMARY_LENGTH))
            generator_target = tf.concat([self.generator_target, end_of_sequence], axis=1)

            if(self.action == 'pretrain'):
                generator_decoder_inputs = tf.concat([begin_of_sequence, self.generator_target], axis=1)

            real_samples = tf.concat([self.real_samples, end_of_sequence], axis=1)

            global_step = tf.Variable(500, name='global_step', trainable=False, dtype=tf.float32)

        with tf.variable_scope('embedding_layer') as scope:

            initializer = tf.contrib.layers.xavier_initializer()
            embeddings = tf.get_variable(name='embeddings', shape=[self.vocab_size, EMBEDDING_DIMENSION], initializer=initializer, trainable=True)
            generator_decoder_embedded_inputs = tf.nn.embedding_lookup(embeddings, generator_decoder_inputs)

        with tf.variable_scope('generator_layer') as scope:

            generator_lstm_length = [ARTICLE_LENGTH + 1 for _ in range(BATCH_SIZE)]

            generator_output_raw, generator_output_ids, generator_ouput_probability, generator_coverage_loss = generator(
                encoder_inputs = self.source,
                vocab_size = self.vocab_size,
                embeddings = embeddings,
                encoder_length = generator_lstm_length,
                decoder_inputs = generator_decoder_embedded_inputs,
                feed_previous = self.action=='pretrain',
                do_sample = self.action=='train',
                do_beam_search= self.action=='test'
            )

            generator_output_ids = tf.stop_gradient(tf.stack(generator_output_ids, axis=1))
            self.generator_prediction = generator_output_ids
            self.log_p = tf.stack(generator_output_raw, axis=1)
            generator_probability = tf.reduce_max(generator_output_raw, axis=-1)
            self.generator_probability = tf.reduce_mean(generator_probability)
            if self.action == 'test':
                self.generator_variables = [v for v in tf.trainable_variables() if v.name.startswith("generator_layer") or v.name.startswith("embedding_layer")]
                return
            
            scope.reuse_variables()

            generator_argmax_outputs, baseline_ids, _ ,_ = generator(
                encoder_inputs = self.source,
                vocab_size = self.vocab_size,
                embeddings = embeddings,
                encoder_length = generator_lstm_length,
                decoder_inputs = generator_decoder_embedded_inputs,
                feed_previous = True,
                do_sample = False
            )

            generator_argmax_outputs = tf.stack(generator_argmax_outputs, axis=1)
            baseline_ids = tf.stop_gradient(tf.stack(baseline_ids, axis=1))

        with tf.variable_scope("discriminator_layer") as scope: 
            true_sample_prediction = tf.nn.sigmoid(discriminator(real_samples, self.summary_lstm_length, self.vocab_size))
            scope.reuse_variables()
            false_sample_prediction = tf.nn.sigmoid(discriminator(generator_output_ids, self.summary_lstm_length, self.vocab_size))

        with tf.variable_scope("reconstructor_layer") as scope:
            reconstructor_sample_loss, _, _ = reconstructor(
                encoder_inputs =  generator_output_ids,
                vocab_size = self.vocab_size,
                encoder_length = self.summary_lstm_length,
                decoder_inputs = decoder_reconstructor_inputs,
                decoder_targets = decoder_reconstructor_targets
            )

            scope.reuse_variables()

            reconstructor_argmax_loss,_,_= reconstructor(
                encoder_inputs = baseline_ids,
                vocab_size = self.vocab_size,
                encoder_length = self.summary_lstm_length,
                decoder_inputs = decoder_reconstructor_inputs,
                decoder_targets = decoder_reconstructor_targets
            )

        self.generator_variables = [v for v in tf.trainable_variables() if v.name.startswith("generator_layer") or v.name.startswith("embedding_layer")]
        self.discriminator_variables = [v for v in tf.trainable_variables() if v.name.startswith("discriminator_layer")]
        self.reconstructor_variables = [v for v in tf.trainable_variables() if v.name.startswith("reconstructor_layer")]

        with tf.variable_scope("generator_layer_loss") as scope:

            scores = []
            length = tf.cast(len(generator_ouput_probability), dtype=tf.float32)

            reconstructor_base = tf.maximum(RECONSTRUCTOR_BASE - global_step * 0.00002, 0.)
            reconstructor_base = 0.0
            reconstructor_weight = tf.minimum(RECONSTRUCTOR_WEIGHT + global_step * 0.00005, 1.2)
            rs = -(tf.stop_gradient(reconstructor_sample_loss) - tf.stop_gradient(reconstructor_argmax_loss)) + reconstructor_base
            reconstructor_score = reconstructor_weight * rs
                    
            for i, cur_total_score in enumerate(batch_to_time_major(false_sample_prediction)):
                if i == 0:
                    score = tf.stop_gradient(cur_total_score)
                else:
                    score = tf.stop_gradient(cur_total_score)
                score = 2. * score - 1
                score = score - tf.reduce_mean(score)
                scores.append(score)
                last_score = tf.stop_gradient(cur_total_score)

            discount_scores = [[]] * len(scores)
            running_add = 0.0
            discount_rate = 0.3

            for i in reversed(range(len(scores))):
                running_add = running_add * discount_rate + scores[i]
                discount_scores[i] = running_add + reconstructor_score

            total_loss = []
            total_coverage_loss = []
            for cur_score, prob, c_l in zip(discount_scores, generator_ouput_probability, generator_coverage_loss):
                loss = tf.reduce_mean(-cur_score * tf.log(tf.clip_by_value(prob, 1e-7, 1.0)))
                one_coverage_loss = COVERAGE_WEIGHT * tf.reduce_mean(tf.reduce_sum(c_l, axis=1))
                loss += one_coverage_loss
                total_coverage_loss.append(one_coverage_loss)
                total_loss.append(loss)

            self.generator_loss = tf.add_n(total_loss)

        with tf.variable_scope("pretrain_generator_loss") as scope:

            generator_target = batch_to_time_major(generator_target)
            total_loss = []
            total_coverage_loss = []
            length = tf.cast(len(generator_target), dtype=tf.float32)
            for prob_t,target,c_l in zip(generator_output_raw, generator_target, generator_coverage_loss):
                target_prob = tf.reduce_max(tf.one_hot(target, self.vocab_size) * prob_t, axis=-1)
                one_coverage_loss = COVERAGE_WEIGHT * tf.reduce_mean(tf.reduce_sum(c_l, axis=1))
                loss = -tf.reduce_mean(tf.log(tf.clip_by_value(target_prob, 1e-9, 1.0))) + 0.1 * one_coverage_loss
                total_coverage_loss.append(one_coverage_loss)
                total_loss.append(loss)
            self.pretrain_generator_loss = tf.add_n(total_loss) / length
            self.pretrain_coverage_loss = tf.add_n(total_coverage_loss) / length

        with tf.variable_scope("discriminator_layer_loss") as scope:

            self.discriminator_loss = -tf.reduce_mean(tf.log(true_sample_prediction + 1e-9)) - tf.reduce_mean(tf.log(1. - false_sample_prediction + 1e-9))

        with tf.variable_scope("reconstruct_layer_loss") as scope:

            self.reconstruct_loss = tf.reduce_mean(reconstructor_argmax_loss)

        with tf.variable_scope("optimizer") as scope:

            self.step_increment_operation = tf.assign(global_step, global_step + 1)

            if self.action =='pretrain':
                generator_pretrain_optimizer = tf.train.AdamOptimizer(0.0001)
                gradients, variables = zip(*generator_pretrain_optimizer.compute_gradients(self.pretrain_generator_loss, var_list=self.generator_variables))
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.generator_pretrain_optimizer = generator_pretrain_optimizer.apply_gradients(zip(gradients, variables))

            if self.action == 'train':
                self.discriminator_train_optimizer = tf.train.AdamOptimizer(0.002, beta1=0.5, beta2=0.999).minimize(
                    self.discriminator_loss, 
                    var_list=self.discriminator_variables
                )
                train_generator_op = tf.train.AdamOptimizer(0.00005, beta1=0.5, beta2=0.999)
                gradients, variables = zip(*train_generator_op.compute_gradients(self.generator_loss, var_list=self.generator_variables))
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.generator_train_optimizer = train_generator_op.apply_gradients(zip(gradients, variables))
                self.reconstructor_train_optimizer = tf.train.AdamOptimizer(0.0001).minimize(
                    self.reconstruct_loss,
                    var_list=self.reconstructor_variables
                )

    def pretrain(self):

        print("Begin Pretraining.")

        step = 0

        if not os.path.exists('./generator'):
            os.makedirs('./generator')
        model_path = './generator/pretrain_model'
        saver = self.generator_saver

        self.sess.run(tf.global_variables_initializer())

        if self.load:
            saver.restore(self.sess, tf.train.latest_checkpoint('./generator'))

        for x_batch, y_batch in self.data.pretrain_generator(self.datatype):

            cur_loss = 0.0
            coverage_loss = 0.0
            cur_probability = 0.0

            step = int(self.sess.run(self.step_increment_operation))

            _, loss, c_loss, _, probability = self.sess.run([
                self.generator_pretrain_optimizer, self.pretrain_generator_loss, self.pretrain_coverage_loss, self.generator_prediction, self.generator_probability
            ], feed_dict = {
                self.source: x_batch,
                self.generator_target: y_batch
            })

            cur_loss += loss
            coverage_loss += c_loss
            cur_probability += probability

            print('Step {step}: G_Loss: {loss} Coverage_Loss: {c_loss} Probability: {probability}'.format(step=step, loss=cur_loss, c_loss=coverage_loss, probability=cur_probability))

            if step % PRETRAIN_CHECKPOINT == 0:
                saver.save(self.sess, model_path, global_step=step)

            if step >= PRETRAIN_NUM_STEPS:
                break

    def train(self):

        print("Begin training.")

        step = 0

        if not os.path.exists('./generator'):
            os.makedirs('./generator')
        generator_model_path = './generator/model'

        if not os.path.exists('./discriminator'):
            os.makedirs('./discriminator')
        discriminator_model_path = './discriminator/model'

        if not os.path.exists('./reconstructor'):
            os.makedirs('./reconstructor')
        reconstructor_model_path = './reconstructor/model'

        self.sess.run(tf.global_variables_initializer())

        self.generator_saver.restore(self.sess, tf.train.latest_checkpoint('./generator'))
        if self.load:
            self.discriminator_saver.restore(self.sess, tf.train.latest_checkpoint('./discriminator'))
            self.reconstructor_saver.restore(self.sess, tf.train.latest_checkpoint('./reconstructor'))

        data_generator = self.data.train_generator(self.datatype)

        for _ in range(TRAIN_NUM_STEPS):

            generator_probability = 0.0
            generator_loss = 0.0
            discriminator_loss = 0.0
            reconstructor_loss = 0.0

            step = int(self.sess.run(self.step_increment_operation))

            for i in range(DISCRIMINATOR_ITERATIONS):

                x_batch, y_batch = data_generator.__next__()
                _, loss = self.sess.run([
                    self.discriminator_train_optimizer, self.discriminator_loss
                ], feed_dict = {
                    self.source: x_batch,
                    self.real_samples: y_batch
                })

                discriminator_loss += loss / DISCRIMINATOR_ITERATIONS

            if step < PRETRAIN_DISCRIMINATOR_RECONSTRUCTOR_STEPS:

                for i in range(RECONSTRUCTOR_ITERATIONS):

                    x_batch, y_batch = data_generator.__next__()
                    _, loss = self.sess.run([
                        self.reconstructor_train_optimizer, self.reconstruct_loss
                    ], feed_dict = {
                        self.source: x_batch,
                        self.decoder_reconstructor_inputs: x_batch
                    })

                    reconstructor_loss += loss / RECONSTRUCTOR_ITERATIONS

            if step >= PRETRAIN_DISCRIMINATOR_RECONSTRUCTOR_STEPS:

                x_batch, y_batch = data_generator.__next__()

                _, _, rec_loss, gen_loss, gen_prob, _ = self.sess.run([
                    self.generator_train_optimizer, self.reconstructor_train_optimizer, self.reconstruct_loss, self.generator_loss, self.generator_probability, self.generator_prediction
                ], feed_dict = {
                    self.source: x_batch,
                    self.decoder_reconstructor_inputs: x_batch
                })

                reconstructor_loss += rec_loss
                generator_loss += gen_loss
                generator_probability += gen_prob

            print('Step {step}: G_LOSS: {g_loss} G_PROB: {g_prob} D_LOSS: {d_loss} R_LOSS: {r_loss}'.format(step=step, g_loss=generator_loss, g_prob=generator_probability, d_loss=discriminator_loss, r_loss=reconstructor_loss))

            if step % TRAIN_CHECKPOINT == 0:
                self.generator_saver.save(self.sess, generator_model_path, global_step=step)
                self.discriminator_saver.save(self.sess, discriminator_model_path, global_step=step)
                self.reconstructor_saver.save(self.sess, reconstructor_model_path, global_step=step)

            if step >= TRAIN_NUM_STEPS:
                break

    def test(self):

        if not os.path.exists('./output'):
            os.makedirs('./output')

        output_file = open('./output/prediction.txt', 'w')

        self.sess.run(tf.global_variables_initializer())

        self.generator_saver.restore(self.sess, tf.train.latest_checkpoint('./generator'))

        progress = 0

        for x_batch in self.data.test_generator():

            prediction, _ = self.sess.run([
                self.generator_prediction, self.generator_probability
            ], feed_dict = {
                self.source: x_batch
            })

            for i in range(len(prediction)):

                end = 0
                output = [[]] * len(prediction[i])

                for j in reversed(range(len(prediction[i]))):

                    output[j] = prediction[i][j][end] % self.vocab_size
                    end = int(prediction[i][j][end] / self.vocab_size)

                output_file.write(self.data.id2sentence(output) + '\n')

            progress += BATCH_SIZE
            print(progress)

        output_file.close()

        output_file = open('./output/prediction.txt', 'r+')
        hypophysis = output_file.readlines()
        references = open('./data/gigaword/test_summary.txt').readlines()

        rouge = Rouge()

        scores = rouge.get_scores(hypophysis, references, avg=True)

        scores_json = json.dumps(scores, indent=2)
        print(scores_json)
        output_file.write('\n\nRouge Scores:\n\n')
        output_file.write(scores_json)

    def save(self):

        self.sess.run(tf.global_variables_initializer())
        self.generator_saver.restore(self.sess, tf.train.latest_checkpoint('./generator'))
            
        builder = tf.saved_model.builder.SavedModelBuilder('./generator/savedmodel')
        builder.add_meta_graph_and_variables(self.sess, [], strip_default_attrs=True, saver=self.generator_saver)
        builder.save()

    def server_init(self):

        self.sess.run(tf.global_variables_initializer())

        tf.saved_model.loader.load(self.sess, [], './generator/savedmodel')

    def input_prediction(self, articles, summaries):

        generated_summaries = []

        cleaned_articles = []

        for article in articles:

            cleaned_articles.append(self.data.remove_stopwords_and_punctuation(article).lower())

        for article in cleaned_articles:

            slices = []

            output = []

            if len(article.split()) <= ARTICLE_LENGTH:

                slices = [article]

            else:

                words = article.split()

                i = 0
                while i < len(words):
                    slice_end = 0
                    if i + ARTICLE_LENGTH >= len(words):
                        slice_end = len(words)
                    else:
                        slice_end = i + ARTICLE_LENGTH
                    slices.append(' '.join(words[i:slice_end]))
                    i += ARTICLE_LENGTH

            for x_batch in self.data.input_generator(slices):

                prediction, _ = self.sess.run([
                    self.generator_prediction, self.generator_probability
                ], feed_dict = {
                    self.source: x_batch
                })

                for i in range(len(prediction)):

                    end = 0
                    ids = [[]] * len(prediction[i])

                    for j in reversed(range(len(prediction[i]))):

                        ids[j] = prediction[i][j][end] % self.vocab_size
                        end = int(prediction[i][j][end] / self.vocab_size)

                    if all(id == 0 for id in ids):
                        continue

                    output.append(self.data.id2sentence(ids))

            generated_summaries.append(' '.join(output))

        rouge = Rouge()
        scores = []

        for i in range(len(articles)):
            if summaries[i] != '':
                scores.append(rouge.get_scores(generated_summaries[i], summaries[i], avg=False)[0])
            else:
                scores.append(None)

        return generated_summaries, scores

def generator(encoder_inputs, vocab_size, embeddings, encoder_length, decoder_inputs, feed_previous, do_sample=False, do_beam_search=False, latent_dim=300):

    if do_beam_search:
        beam_size = 5
        batch_size = encoder_inputs.get_shape().as_list()[0]
        path_probs = tf.concat([tf.zeros([batch_size,1]), tf.zeros([batch_size,beam_size-1])-10.], axis=1)
        batch_start_id = tf.reshape(tf.tile(tf.reshape(tf.range(batch_size), [-1,1]),[1,beam_size]), [-1]) * beam_size
        encoder_inputs = tf.reshape(tf.tile(encoder_inputs, [1, beam_size]), [batch_size * beam_size, -1])
        encoder_length = encoder_length * beam_size
        penalty1 = np.ones([batch_size, beam_size, vocab_size])
        penalty1[:, :, 0] = 0.1
        penalty2 = np.ones([batch_size, beam_size, vocab_size])
        penalty2[:, :, 0] = 0.5
        penalty3 = np.ones([batch_size, beam_size, vocab_size])

    input_one_hot = tf.one_hot(encoder_inputs,vocab_size)
    encoder_embedded_inputs = tf.nn.embedding_lookup(embeddings, encoder_inputs)
    encoder_shape = encoder_inputs.get_shape().as_list()

    decoder_inputs = batch_to_time_major(decoder_inputs)
    if do_beam_search:
        bos_input = tf.reshape(tf.tile(decoder_inputs[0],[1,beam_size]),[batch_size*beam_size,-1])

    with tf.variable_scope("generator_encoder") as scope:
        cell = tf.contrib.rnn.LSTMCell(num_units=latent_dim*2, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        encoder_outputs,encoder_state = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32, sequence_length=encoder_length, inputs=encoder_embedded_inputs)

    with tf.variable_scope("generator_pointer_decoder") as scope:

        V = tf.get_variable(name="V", shape=[latent_dim, 1])
        W_h = tf.get_variable(name="W_h", shape=[latent_dim * 2, latent_dim])
        W_s = tf.get_variable(name="W_s", shape=[latent_dim * 2, latent_dim])
        b_attn = tf.get_variable(name="b_attn", shape=[latent_dim])
        w_c = tf.get_variable(name="w_c", shape=[latent_dim])

        cell = tf.contrib.rnn.LSTMCell(num_units=latent_dim*2, state_is_tuple=True)

        def input_projection(raw_input, last_attention_context):
            return tf.layers.dense(tf.concat([raw_input, last_attention_context], axis=1), latent_dim * 2, name="input_projection")

        def do_attention(state,c_t):
            e_t = []
            attention_state = encoder_outputs
            c_t = tf.split(c_t,num_or_size_splits=encoder_shape[1],axis=1)

            for h_i,c_i in zip(batch_to_time_major(attention_state),c_t):
                hidden = tf.tanh(tf.matmul(h_i,W_h) + tf.matmul(state,W_s) + w_c*c_i + b_attn)
                e_t_i = tf.squeeze(tf.matmul(hidden,V),1)
                e_t.append(e_t_i)

            attention_weight = tf.nn.softmax(tf.stack(e_t,axis=1))
            attention_context = tf.squeeze(tf.matmul(tf.expand_dims(attention_weight,axis=1),attention_state),axis=1)
            return attention_weight,attention_context

        def get_pointer_distribution(attention_weight):
            return tf.squeeze(tf.matmul(tf.expand_dims(attention_weight,axis=1),input_one_hot),axis=1)

        def get_vocab_distribution(state,attention_context):
            hidden = tf.layers.dense(tf.concat([state,attention_context],axis=1),1000,name='P_vocab_projection1')
            vocab_weight = tf.layers.dense(hidden,vocab_size,name='P_vocab_projection2')
            return tf.nn.softmax(vocab_weight)

        output_raw = []
        output_ids = []
        output_probability = []
        coverage_loss = []

        state = encoder_state
        attention_coverage = tf.zeros([encoder_shape[0],encoder_shape[1]])
        attention_weight,attention_context = do_attention(state.h,attention_coverage)

        for i in range(len(decoder_inputs)):
            if i > 0:
                scope.reuse_variables()
            if i == 0 and do_beam_search:
                input_t = bos_input
            elif i > 0 and do_beam_search:
                if i <= 7:
                    penalty = np.log(penalty1)
                elif i <= 10:
                    penalty = np.log(penalty2)
                else:
                    penalty = np.log(penalty3)
                
                out_probs = tf.reshape(output_raw[-1], [batch_size,beam_size, vocab_size])
                new_path_prob = tf.reshape(path_probs,[batch_size,beam_size,1]) + tf.log(out_probs + 1e-7) + penalty
                new_path_prob = tf.reshape(new_path_prob, [batch_size, beam_size*vocab_size])
                path_probs,top_ids = tf.nn.top_k(new_path_prob, beam_size)
                output_ids.append(top_ids)
                output_probability.append(path_probs)
                top_batch_ids = batch_start_id + tf.floordiv(tf.reshape(top_ids,[batch_size*beam_size]), vocab_size)
                new_s_c,new_s_h = tf.gather(state.c,top_batch_ids),tf.gather(state.h, top_batch_ids)
                state = tf.nn.rnn_cell.LSTMStateTuple(new_s_c, new_s_h)
                attention_coverage = tf.gather(attention_coverage, top_batch_ids)
                attention_context = tf.gather(attention_context, top_batch_ids)
                last_output_id = tf.reshape(tf.mod(top_ids,int(vocab_size)),[batch_size*beam_size])
                input_t = tf.nn.embedding_lookup(embeddings,last_output_id)

            elif i > 0 and feed_previous and do_sample:
                last_output_id = sample2D(output_raw[-1])
                batch_id = tf.cast(tf.range(encoder_shape[0]),dtype=tf.int64)
                abs_id = tf.concat([tf.expand_dims(batch_id,axis=1),tf.expand_dims(last_output_id,axis=1)],axis=1)
                last_output_prob = tf.gather_nd(output_raw[-1],abs_id)
                input_t = tf.nn.embedding_lookup(embeddings,last_output_id)

            elif i > 0 and feed_previous:
                last_output_id = tf.argmax(output_raw[-1],axis=-1)
                batch_id = tf.range(encoder_shape[0])
                abs_id = tf.concat([tf.expand_dims(last_output_id,axis=1),tf.expand_dims(last_output_id,axis=1)],axis=1)
                last_output_prob = tf.gather_nd(output_raw[-1],abs_id)
                input_t = tf.nn.embedding_lookup(embeddings,last_output_id)
            else:
                input_t = decoder_inputs[i]
            if feed_previous and i > 0 and not do_beam_search:
                output_ids.append(last_output_id)
                output_probability.append(last_output_prob)

            _, state = cell(input_projection(input_t,attention_context),state)
            attention_weight, attention_context = do_attention(state.h,attention_coverage)
            coverage_loss.append(tf.minimum(attention_coverage,attention_weight))
            attention_coverage += attention_weight

            P_gen = tf.sigmoid(tf.layers.dense( tf.concat([input_t,state.h,attention_context], axis=1), 1, name='P_gen'))
            output_t = P_gen*get_vocab_distribution(state.h,attention_context) + (1 - P_gen)*get_pointer_distribution(attention_weight)    
            output_raw.append(output_t)

        if do_beam_search:
            out_probs = tf.reshape(output_raw[-1], [batch_size,beam_size,vocab_size])
            new_path_prob = tf.reshape(path_probs,[batch_size,beam_size,1]) + tf.log(out_probs+1e-7)
            new_path_prob = tf.reshape(new_path_prob,[batch_size, beam_size*vocab_size])
            path_probs,top_ids = tf.nn.top_k(new_path_prob, beam_size)
            output_ids.append(top_ids)
            output_probability.append(path_probs)
        elif feed_previous:
            output_ids.append(tf.argmax(output_raw[-1],axis=-1))
            output_probability.append(tf.reduce_max(output_raw[-1],axis=1))
        else:
            output_ids = [tf.argmax(output,axis=-1) for output in output_raw]
            output_probability = [tf.reduce_max(output,axis=-1) for output in output_raw]

    return output_raw, output_ids, output_probability, coverage_loss

def discriminator(inputs, lstm_length, vocab_size):
    
    with tf.variable_scope("discriminator_word_embedding") as scope:
        initializer = tf.contrib.layers.xavier_initializer()
        discriminator_embeddings = tf.get_variable(name="embeddings", shape=[vocab_size, 300], initializer=initializer, trainable = True)
        inputs = tf.nn.embedding_lookup(discriminator_embeddings, inputs)
    
    cell = tf.contrib.rnn.LSTMCell(num_units=500, state_is_tuple=True)
    lstm_outputs, last_states = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32, sequence_length=lstm_length, inputs=inputs)

    with tf.variable_scope("output_project") as scope:
        outputs = tf.contrib.layers.linear(lstm_outputs, 1, scope=scope)
    
    return tf.squeeze(outputs, axis=2)

def reconstructor(encoder_inputs, vocab_size, encoder_length, decoder_inputs, decoder_targets, latent_dim=200):

    initializer = tf.contrib.layers.xavier_initializer()

    embeddings = tf.get_variable(
        name="reconstructor_embeddings",
        shape=[vocab_size, 100],
        initializer=initializer,   
        trainable = True
    )

    decoder_inputs = tf.nn.embedding_lookup(embeddings, decoder_inputs)
    input_one_hot = tf.one_hot(encoder_inputs, vocab_size)

    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
    encoder_shape = encoder_inputs.get_shape().as_list()

    decoder_inputs = batch_to_time_major(decoder_inputs)

    with tf.variable_scope("reconstructor_encoder") as scope:
        fw_cell = tf.contrib.rnn.LSTMCell(num_units=latent_dim, state_is_tuple=True)
        bw_cell = tf.contrib.rnn.LSTMCell(num_units=latent_dim, state_is_tuple=True)

        encoder_outputs, state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = fw_cell,
            cell_bw = bw_cell,
            dtype = tf.float32,
            sequence_length = encoder_length,
            inputs = encoder_inputs_embedded,
            time_major = False
        )

        output_fw, output_bw = encoder_outputs
        state_fw, state_bw = state
        encoder_outputs = tf.concat([output_fw,output_bw], 2)
        encoder_state_c = tf.concat((state_fw.c, state_bw.c), 1)
        encoder_state_h = tf.concat((state_fw.h, state_bw.h), 1)
        encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

    with tf.variable_scope("reconstructor_decoder") as scope:

        V = tf.get_variable(name="V", shape=[latent_dim, 1])
        W_h = tf.get_variable(name="W_h", shape=[latent_dim * 2, latent_dim])
        W_s = tf.get_variable(name="W_s", shape=[latent_dim * 2, latent_dim])
        b_attn = tf.get_variable(name="b_attn", shape=[latent_dim])
        w_c = tf.get_variable(name="w_c", shape=[latent_dim])

        cell = tf.contrib.rnn.LSTMCell(num_units=latent_dim*2, state_is_tuple=True)

        def input_projection(raw_input, last_attention_context):
            return tf.layers.dense(tf.concat([raw_input, last_attention_context], axis=1), latent_dim*2, name="input_projection")

        def do_attention(state,c_t):
            e_t = []
            attention_state = encoder_outputs
            c_t = tf.split(c_t,num_or_size_splits=encoder_shape[1],axis=1)

            for h_i,c_i in zip(batch_to_time_major(attention_state),c_t):
                hidden = tf.tanh(tf.matmul(h_i,W_h) + tf.matmul(state,W_s) + w_c*c_i + b_attn)
                e_t_i = tf.squeeze(tf.matmul(hidden,V),1)
                e_t.append(e_t_i)

            attention_weight = tf.nn.softmax(tf.stack(e_t,axis=1))
            attention_context = tf.squeeze(tf.matmul(tf.expand_dims(attention_weight,axis=1),attention_state),axis=1)
            return attention_weight,attention_context

        def get_pointer_distribution(attention_weight):
            return tf.squeeze(tf.matmul(tf.expand_dims(attention_weight,axis=1),input_one_hot),axis=1)

        def get_vocab_distribution(state,attention_context):
            hidden = tf.layers.dense(tf.concat([state,attention_context],axis=1),500,name='P_vocab_projection1')
            vocab_weight = tf.layers.dense(hidden,vocab_size,name='P_vocab_projection2')
            return tf.nn.softmax(vocab_weight)

        decoder_outputs = []
        state = encoder_state
        attention_coverage = tf.zeros([encoder_shape[0],encoder_shape[1]])
        attention_weight,attention_context = do_attention(state.h,attention_coverage)

        for i in range(len(decoder_inputs)):
            if i > 0:
                scope.reuse_variables()
            input_t = decoder_inputs[i]

            _, state = cell(input_projection(input_t,attention_context),state)
            attention_weight,attention_context = do_attention(state.h,attention_coverage)
            attention_coverage += attention_weight

            P_gen = tf.sigmoid(tf.layers.dense( tf.concat([input_t,state.h,attention_context], axis=1), 1, name='P_gen'))
            output_t = P_gen*get_vocab_distribution(state.h,attention_context) + (1 - P_gen)*get_pointer_distribution(attention_weight)         
            decoder_outputs.append(output_t)

    attention_coverage = tf.split(attention_coverage,num_or_size_splits=encoder_shape[1],axis=1)

    with tf.variable_scope("reconstructor_loss") as scope:
        targets = batch_to_time_major(decoder_targets)
        total_loss = []
        for prob_t,target in zip(decoder_outputs,targets):
            target_prob = tf.reduce_max(tf.one_hot(target,vocab_size)*prob_t,axis=-1)
            cross_entropy = -tf.log(tf.clip_by_value(target_prob,1e-10,1.0))
            total_loss.append(cross_entropy)
        total_loss = tf.reshape(tf.reduce_mean(tf.stack(total_loss,axis=1),axis=1),[-1])

    return total_loss, decoder_outputs, attention_coverage