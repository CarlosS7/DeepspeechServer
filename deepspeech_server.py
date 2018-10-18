#!/usr/bin/env python
#* -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import wave
import sys
import flask
import io
import sentimentanalysis

from deepspeech import Model, printVersions
from timeit import default_timer as timer



# These constants control the beam search decoder
# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

#The alpha hyperparameter of the CTC decoder. Language Model Weight
LM_WEIGHT = 1.50

#Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
VALID_WORD_COUNT_WEIGHT = 2.25

# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the firt layer), so make sure you use the same constants that
# were used during training

#Number of MFCC feature to use
N_FEATURES = 26

# size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9

#Model
model = 'models/output_graph.pbmm'
alphabet = 'models/alphabet.txt'
lm = 'models/lm.binary'
trie = 'models/trie'

# initialize our Flask application
app = flask.Flask(__name__)

@app.route("/transcript", methods=["POST"])
def main():

    #initialize the data dictionary that will be returned from the 
    #view
    data = {"success": False}
    # ensure that an audio file was properly uploadec to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("audio"):
            fin = wave.open(flask.request.files["audio"], 'rb')
            fs = fin.getframerate()
            audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
            audio_length = fin.getnframes() * (1/16000)
            fin.close()

            print('Loading model from file', file = sys.stderr)
            model_load_start = timer()
            ds = Model(model, N_FEATURES, N_CONTEXT, alphabet, BEAM_WIDTH)
            model_load_end = timer() - model_load_start
            print('Loaded model in {:.3}s.'.format(model_load_end), file=sys.stderr)
            lm_load_start = timer()
            ds.enableDecoderWithLM(alphabet, lm, trie, LM_WEIGHT, VALID_WORD_COUNT_WEIGHT)
            lm_load_end = timer() - lm_load_start
            print('Loaded model in {:.3}s.'.format(lm_load_end), file=sys.stderr)

            print('Running inference.', file = sys.stderr)
            inference_start = timer()
            text = ds.stt(audio, fs)
            inference_end = timer() - inference_start
            print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file = sys.stderr)
            
            data["results"] =[]

            data["results"].append(text)

            data["success"] = True

            data["sentiment"] = sentimentanalysis.get_score(text)
            
            return flask.jsonify(data)

if __name__ == '__main__':
    print(('* Loading the Deepspeech server model and Flask starting server...'
        'please wait until server has fully started'))
    app.run(host='0.0.0.0', debug=True)



