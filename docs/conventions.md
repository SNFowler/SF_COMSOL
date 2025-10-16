# Recording Data

For each individual evaluation during the training or sweep process, data is to be saved as json objects in a file with the form *.jsonl

The experimental config, the elements which are not changing during the experiment is saved in the same folder as each run, in the form

""" AnalysisTools.ExperimentRunner.py
    def save_results(self, results, filename):
        """Save list of results to jsonl"""
        with open(self.path(filename), "w") as f:
            for result in results:
                f.write(json.dumps(result, default=self._json_convert) + "\n")
                
    def save_config(self, config, filename="config.json"):
        """Save config dict to json"""
        with open(self.path(filename), "w") as f:
            json.dump(config, f, indent=2, default=self._json_convert)
"""

