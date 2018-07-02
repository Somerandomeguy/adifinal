import os
import json
import pandas as pd
import numpy as np
import scipy
import scipy.stats
import requests
from encoder import Model
from flask import (Flask,
    flash, g, redirect, render_template, request, url_for, current_app as app, jsonify, make_response
)

app = Flask(__name__)
app.config.from_mapping(
    SECRET_KEY='dev'
)

app.config.from_pyfile('config.py', silent=True)


@app.route('/api/content', methods=['POST'])
def content():
    data = request.get_json()
    interest = data['queryResult']['intent']['displayName']
    raw_text = data['queryResult']['queryText']
    dataanalyzer = UserData(raw_text)
    data = dataanalyzer.get_intent(interest, 'Bob')
    return make_response(jsonify({'fulfillmentText': data}))



class UserData():


    def __init__(self, rawtext):
        #external data source is a todo
        self.df = pd.read_csv("sample_data.csv")
        #must be converted to series
        self.recent_run = self.df[self.df['End Timestamp'] == self.df['End Timestamp'].max()].iloc[0]
        self.summaries = ['default']
        self.hard_attributes = ['Max. Elevation', 'Average Heart Rate', 'Calories', 'Elevation Gain', 'Elevation Loss']
        self.how_good = ['Average Moving Speed', 'Moving Duration (h:m:s)', 'Temperature (Raw)']
        self.custom_feedbacks = ['fb_compare_shoes']
        self.rawtext = rawtext



    def get_intent(self, intent, userName):
        if intent in self.summaries:
            return self._ret_summary(intent)
        elif intent in self.hard_attributes:
            return self._ret_value(intent)
        elif intent in self.how_good:
            return self._compare(intent)
        elif intent in self.custom_feedbacks:
            return self._get_feedback(intent)
        else:
            return "I could not find an answer"


    def _ret_summary(self, intent):
        heartbeat = self.recent_run['Average Heart Rate (bpm)']
        avg_speed = self._compare('Average Moving Speed')
        calories = self.recent_run['Calories']
        return "Your performance was {} with an average heart rate of {}. You burned {} calories".format(heartbeat, avg_speed, calories)



    def _compare(self, intent):
        res = scipy.stats.percentileofscore(self.df[intent], self.recent_run[intent])
        if res > 0.95:
            return "extraordinary"
        elif res > 0.8:
            return "very good"
        elif res > 0.5:
            return "good"
        elif res > 0.3:
            return "not the best"
        else:
            return "terrible"


    def _ret_value(self, val):
        return self.recent_run[val]

    def _progress(self, val):
        #TODO
        return True


    def _get_feedback(self, val):
        #model = Model()
        #text = [self.rawtext]
        #features = model.transform(text)
        #TODO save features to database to extrapolate features and train the next classifier on the weights
        return "Thank you very much for the feedback"









