from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import textwrap
from Llama import LlamaInference
import pandas as pd
import os
from Graph_description import GraphDescriptionPipeline
# from ipywidgets import interact, widgets
import os
from deepseeklm import DeepSeekLM
import json
import pandas as pd
from Llama import LlamaInference
from DataAquisition import DataAquisition
# from realtime_analysis import RealtimeAnalysis
from IPython.display import clear_output
from IPython.display import JSON
from Graph_description import GraphDescriptionPipeline
import mercury as mr
from CodeGeneration import CodeGeneration
from Inference_Classification import Inference_Classification
from Inference_Regression import Inference_Regression


# class DropDown:
#     def __init__(self,Knowledge_base):
#         # Specify the folder path
#         self.Knowledge_base = Knowledge_base
#         self.problem_type = None
#         self.source = None
#         self.folder_inside_source = None
        
#         self.problem_types = None
#         self.problem_type_dropdown = None
#         self.source_dropdown = None
#         self.folder_inside_source_dropdown = None
#         self.selected_problem_type = None
#         self.selected_source = None
        

#     # Function to get the list of directories
#     def get_directory_list(self,folder_path='.'):
#         return [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
   
#     def get_values(self):
#         return self.problem_type, self.source, self.folder_inside_source

#     def printvalues(self):
#         print(f"Selected Problem Type: {self.problem_type}")
#         print(f"Selected Source: {self.source}")
#         print(f"Selected Folder inside Source: {self.folder_inside_source}")

#     # Function to update dropdown values and run Analyzer
#     def update_dropdown_run_analyzer(self):
#     #     global problem_type, source, folder_inside_source
#         self.problem_type = self.problem_type_dropdown.value
#         self.source = self.source_dropdown.value
#         self.folder_inside_source = self.folder_inside_source_dropdown.value
        
#         # Example: Print the selected problem type, source, and folder inside source
#         print(f"Selected Problem Type: {self.problem_type}")
#         print(f"Selected Source: {self.source}")
#         print(f"Selected Folder inside Source: {self.folder_inside_source}")
        
#         path = f"{self.Knowledge_base}/{self.problem_type}/{self.source}/{self.folder_inside_source}/dataset/{self.folder_inside_source}.csv"
#         df = pd.read_csv(path)
#         auth_token = "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"
#         json_path = f"{self.Knowledge_base}/{self.problem_type}/{self.source}/{self.folder_inside_source}/json/file_paths.json"
#         with open(json_path) as f:
#             json_file = json.load(f)
    
#         analyzer = RealtimeAnalysis(auth_token ,df , self.folder_inside_source, self.problem_type, json_file)
#         print("\n___ Visualization of first few rows of your data ___")
#         display(df.head())
#         print(" \n___ The following data is available for the file you selected please, write your query Accordingly ___ \n")
#         mr.JSON(json_file)
        
#         while True:
#             try:
#                 clear_output(wait=True)
#                 query = input("Write your query & Enter 'exit' to end : ")

#                 if query.lower() == 'exit':
#                     break  # exit the loop if the user enters 'exit'
#                 else:
#                     analyzer.run_analyzer(query)
#             except Exception as e:
#                 # Code to handle the exception
#                 print(f"An exception occurred: {e}")

#         print("Chat ended!")

#     # Attach observer to update the source options when problem type changes
#     def update_sources(self,change):
#         self.selected_problem_type = self.problem_type_dropdown.value
#         self.source_dropdown.options = self.get_directory_list(folder_path = os.path.join(self.Knowledge_base, self.selected_problem_type))
#         self.source_dropdown.value = None
#         self.folder_inside_source_dropdown.options = []
#         self.folder_inside_source_dropdown.value = None

#     # Attach observer to update sources when problem type changes
#     def update_folders(self,change):
#         self.selected_source = self.source_dropdown.value
#         self.selected_problem_type = self.problem_type_dropdown.value
#         self.folder_inside_source_dropdown.options = self.get_directory_list(os.path.join(self.Knowledge_base, self.selected_problem_type, self.selected_source))
#         self.folder_inside_source_dropdown.value = None

#     def run_dropdown(self):
#         # Get the list of directories for problem types
#         self.problem_types = self.get_directory_list(self.Knowledge_base)
#         # Create a dropdown widget for problem types
#         self.problem_type_dropdown = Dropdown(options=self.problem_types, description='Select your Problem Type')
#         self.problem_type_dropdown.observe(self.update_sources, names='value')
#         # Get the list of directories for sources based on the selected problem type
#         self.source_dropdown = Dropdown(options=[], description='Select your Source')
#         self.source_dropdown.observe(self.update_folders, names='value')
#         # Create a dropdown widget for folders inside the selected source
#         self.folder_inside_source_dropdown = Dropdown(options=[], description='Select Folder inside Source')
#         # Display the widgets
#         display(self.problem_type_dropdown)
#         display(self.source_dropdown)
#         display(self.folder_inside_source_dropdown)
#         # Create a button to manually trigger the update
#         interact_manual(self.update_dropdown_run_analyzer)



class RealtimeAnalysis:
    def __init__(self, auth_token, df, file_name, problem_type, source, json_file):
        self.auth_token = auth_token
        self.llama_inference = LlamaInference(self.auth_token)
        # Graph Description
        self.img_to_text = GraphDescriptionPipeline()
      
        self.file_name = file_name
        self.data = df
        self.problem_type = problem_type
        self.json_file = json_file
        self.fn_list = []
        self.dependent_var = None
        self.source = source

        if self.problem_type == "time series":
            self.fn_list = ['Introduction of the dataset','Summary statistics explainer','Domain Explainer', 'Probability distribution visualization', 'Missing Number plot visualization', 'Heatmap of dataset', 'correlation matrix explainer', 'Training loss visualization','Most Correlated Features explainer','NO function matches found form list']
        
        elif self.problem_type == "categorical":
            self.fn_list = ['Introduction of the dataset','Summary statistics explainer','Domain Explainer','Crosstab of two variables','Chi Square Statistics for relationship','Bar chart visualization', 'Two variable histogram visualization', 'Probability distribution visualization', 'Missing Number plot visualization','Two variable cross tabulation chart', 'Heatmap of dataset', 'Confusion matrix explainer', 'inference from classification model', 'NO function matches found form list']
        
        elif self.problem_type == "numerical":
            self.fn_list = ['Introduction of the dataset','Summary statistics explainer','Domain Explainer','Bar chart visualization', 'Two variable histogram visualization', 'Probability distribution visualization', 'Missing Number plot visualization', 'Heatmap of dataset', 'Pairwise plot explainer','Most_Correlated_Features_explainer_csv', 'inference from regression model', 'NO function matches found form list']
        
    def run_analyzer(self, query):
        
        fn_found = self.llama_inference.function_caller(self.fn_list, query)
        print("The suggested function is : ", fn_found)
        if fn_found == 'NO function matches found form list' or 'No' in fn_found:
            print("No function found relevant to your Query\n")
            print("Loading DeepSeek Coder LM...")
            code_generation = CodeGeneration(query, self.data)
            code_generation.run_process()
        if fn_found == 'correlation matrix explainer':
            self.corrmat_explainer(query)
        elif fn_found == 'Summary statistics explainer':
            self.summary_stats_explainer(query)
        elif fn_found == 'Domain Explainer':
            self.analyze_user_question(query)
        elif fn_found == 'Most Correlated Features explainer':
            self.Most_Corr_Features_explainer(query)
        elif fn_found == 'Training loss Explainer':
            self.training_loss_explainer(query)            
        elif fn_found == 'Bar chart visualization':
            self.bar_chart_visualization(query)
        elif fn_found == 'Two variable histogram visualization':
            self.histogram_visualization(query)
#         elif fn_found == 'Two variable Stacked column chart':
#             self.stacked_column_chart(query)
        elif fn_found == 'Probability distribution visualization':
            self.prob_dist_visualization(query)           
        elif fn_found == 'Missing Number plot visualization':
            self.missing_num_plot_explainer(query)
        elif fn_found == 'Two variable cross tabulation chart':
            self.CrossTabulation_explainer(query)
        elif fn_found == 'Heatmap of dataset':
            self.heatmap_explainer(query)       
        elif fn_found == 'Introduction of the dataset':
            self.llama_inference.dataset_intoduction(self.data)
        elif fn_found == 'Chi Square Statistics for relationship':
            self.chi_stats_explainer(query)
        elif fn_found == 'Crosstab of two variables':
            self.crosstab_explainer(query)
        elif fn_found == 'Confusion matrix explainer':
            self.confusion_matrix_explainer(query)
        elif fn_found == 'Pairwise plot explainer':
            self.Pairplot_explainer(query)
        elif fn_found =='Most_Correlated_Features_explainer_csv':
            self.corrmat_feature_explainer_csv(query)
        elif fn_found == 'inference from regression model':
            self.inference_regression()
        elif fn_found == 'inference from classification model':
            self.inference_classification()
        else:
            print("No function found relevant to your Query\n")
            print("Loading DeepSeek Coder LM...")
            code_generation = CodeGeneration(query, self.data)
            code_generation.run_process()
            
    def analyzer_main(self):
        
        while True:
            try:
                clear_output(wait=True)
                query = input("Write your query & Enter 'exit' to end : ")

                if query.lower() == 'exit':
                    break  # exit the loop if the user enters 'exit'
                else:
                    self.run_analyzer(query)
            except Exception as e:
                # Code to handle the exception
                print(f"An exception occurred: {e}")

        print("Chat ended!")
            
            
# Llama Functions by Hamza

    def corrmat_feature_explainer_csv(self, query):
            value = 'Most correlated Features with dependent Variable '
            found_path = self.json_file.get(value)

            if found_path is None:
                found_path = input("Failed to determine the path. Please enter the path of your relevant file: ")

#             dependent_variable = input("Enter the target variable: ")
#             print(dependent_variable)


            corrmat = pd.read_csv(found_path)
            dependent_variable = self.llama_inference.inference_label_col(corrmat, query)
            print("Dependent_variable: ", dependent_variable)
            print("The path found:", found_path)
            response = self.llama_inference.imp_features(corrmat, dependent_variable)
            print(response)
            
    def corrmat_explainer(self, query):
        dependent_variable = self.dependent_var
        value = 'Most correlated Features with dependent Variable '
        found_path = self.json_file[value]

    #         found_path = self.llama_inference.file_path_extractor(self.json_file , query)
        print("The path found : ", found_path)
        if found_path is None:
            found_path = input("Failed to determine the path please enter path of your relevant file : ")

        dependent_variable = self.llama_inference.inference_label_col(self.data, query)
        print("Dependent_variable: ", dependent_variable)
        if dependent_variable == []:
            dependent_variable = [input("Model fails to recognize your dependent variable, please specify correct variable name")]
        corrmat = pd.read_csv(found_path)
        response = self.llama_inference.corrmat_explain(corrmat, dependent_variable)
        print(response)
        return response
        
    def summary_stats_explainer(self, query):
        value = 'Summary Statistics '
        found_path = self.json_file[value]
#         found_path = self.llama_inference.file_path_extractor(self.json_file , query)
        print("The path found : ", found_path)
        if found_path is None:
            found_path = input("Failed to determine the path please enter path of your relevant file : ")
        stats = pd.read_csv(found_path)
        print("Summary statistics", stats)
        response = self.llama_inference.description(stats,query)
        print(response)
        return response
        
        
    def analyze_user_question(self, question,user_specified_domain='any'):
        try:
            if not user_specified_domain:
                user_specified_domain = "any"
            # Additional processing or analysis logic can be added here
            analysis_result = self.llama_inference.user_domain_question(self.data, question,user_specified_domain)
            print(analysis_result)
            return analysis_result
        except FileNotFoundError as e:
            return f"File not found: {e}"
        except Exception as e:
            return f"An error occurred: {e}"
        
    def chi_stats_explainer(self, query):
        var_found = self.llama_inference.varaible_selector(self.data, query)
        print(f"Explaining chi Square Stats .. ")
        found_path = self.llama_inference.file_path_extractor(self.json_file , query)
        chi_table = pd.read_csv(found_path)
        print("Chi-Square Table : \n ", chi_table)
        if len(var_found) >1:
            self.llama_inference.chi_square(chi_table,query,var1 = var_found[0],var2 = var_found[1])
        else:
            self.llama_inference.chi_square(chi_table,query)

    def crosstab_explainer(self,query):
        var_found = self.llama_inference.varaible_selector(self.data, query)
        print(f"Showing the Cross Tabulation for varaibles {var_found[0]} and {var_found[1]}")
        found_path = self.llama_inference.file_path_extractor(self.json_file , query)      
        if os.path.isfile(f"{found_path}_{var_found[0]}_vs_{var_found[1]}.csv"):
            full_path = f"{found_path}_{var_found[0]}_vs_{var_found[1]}.csv"
        elif os.path.isfile(f"{found_path}_{var_found[1]}_vs_{var_found[0]}.csv"):
            full_path = f"{found_path}_{var_found[1]}_vs_{var_found[0]}.csv"
        else:
            full_path = None
            print("Sorry ! No relevant file found.")
                

        print("File found for your query : ",full_path)
        crosstab = pd.read_csv(found_path)
        response = self.llama_inference.cross_tab(crosstab,query)
        print(response)         
            
# Graph description codes

    def training_loss_explainer(self, query):
        prompt = "USER: <image>\n The given Image is a training loss curve of a machine learning model, You are required to   \nASSISTANT:"
        found_path = self.llama_inference.file_path_extractor(self.json_file , query)
        self.img_to_text.display_image(found_path)
        print("File found for your query : ",found_path)
        response = self.img_to_text.process(found_path,prompt)
        start_index = response.find("ASSISTANT:") + len("ASSISTANT:")
        extracted_text = response[start_index:].strip()
        print(extracted_text)
        return extracted_text
    
    def bar_chart_visualization(self, query):
        var_found = self.llama_inference.varaible_selector(self.data, query)
        print("Showing the Bar chart for varaible : ",var_found)
        value = 'Bar charts '
        found_path = self.json_file[value]
#         found_path = self.llama_inference.file_path_extractor(self.json_file, query)      
        img_path = f"{found_path}_{var_found[0]}.png"
        print("File found for your query : ",img_path)
        self.img_to_text.display_image(img_path)
        response = self.img_to_text.BarCharts(img_path,query,var_found[0])
        print(response)
        
    def histogram_visualization(self, query):
        var_found = self.llama_inference.varaible_selector(self.data, query)
        print(f"Showing the Bar chart for varaible {var_found[0]} and {var_found[1]}")
        value = 'Histogram of two columns '
        found_path = self.json_file[value]
#         found_path = self.llama_inference.file_path_extractor(self.json_file , query)      
        if os.path.isfile(f"{found_path}_{var_found[0]}_vs_{var_found[1]}.png"):
            full_path = f"{found_path}_{var_found[0]}_vs_{var_found[1]}.png"
        elif os.path.isfile(f"{found_path}_{var_found[1]}_vs_{var_found[0]}.png"):
            full_path = f"{found_path}_{var_found[1]}_vs_{var_found[0]}.png"
        else:
            full_path = input("Failed to get your path please write path manually")
            
                
        print(f"{found_path}_{var_found[0]}_vs_{var_found[1]}.png")
        print(f"{found_path}_{var_found[1]}_vs_{var_found[0]}.png")
        print("File found for your query : ",full_path)
        self.img_to_text.display_image(full_path)
        response = self.img_to_text.Histogram(full_path,query,var_found[0], var_found[1])
        print(response)
        
    def stacked_column_chart(self, query):
        var_found = self.llama_inference.varaible_selector(self.data, query)
        print(f"Showing the Stacked Bar chart for varaible {var_found[0]} and {var_found[1]}")
        found_path = self.llama_inference.file_path_extractor(self.json_file , query)      
        if os.path.isfile(f"{found_path}_{var_found[0]}_vs_{var_found[1]}.png"):
              full_path = f"{found_path}_{var_found[0]}_vs_{var_found[1]}.png"
        elif os.path.isfile(f"{found_path}_{var_found[1]}_vs_{var_found[0]}.png"):
              full_path = f"{found_path}_{var_found[1]}_vs_{var_found[0]}.png"
        else:
              full_path = None

        print("File found for your query : ",full_path)
        self.img_to_text.display_image(full_path)
        response = self.img_to_text.StackedBarChart(full_path,query,var_found[0], var_found[1])
        print(response)
        
    def prob_dist_visualization(self,query):
        print(f"Showing the Probability distribution of {self.file_name}")
        value = 'Probability distributions '
        found_path = self.json_file[value]
#         found_path = self.llama_inference.file_path_extractor(self.json_file , query)
        self.img_to_text.display_image(found_path)
        response = self.img_to_text.Distributions(found_path, query)
        print(response)
        
    def heatmap_explainer(self,query):
        print(f"Showing the Heatmap of {self.file_name}")
        found_path = self.llama_inference.file_path_extractor(self.json_file , query)
        self.img_to_text.display_image(found_path)
        response = self.img_to_text.Heatmap_explainer(found_path, query)
        print(response)
        
    def missing_num_plot_explainer(self,query):
        print(f"Showing the Missing num plot of {self.file_name} before cleaning")
        found_path = self.llama_inference.file_path_extractor(self.json_file , query)
        self.img_to_text.display_image(found_path)
        response = self.img_to_text.Missing_Number(found_path, query)
        print(response)
    
    def Most_Corr_Features_explainer(self,query):
        print(f"Showing the most correlated Features with dependent variable")
        value = self.json['Most correlated Features with dependent Variable ']
        found_path = self.llama_inference.file_path_extractor(self.json_file , query)
        self.img_to_text.display_image(found_path)
        response = self.img_to_text.Most_Correlated_Features(found_path, query)
        print(response)
        
    def Trend_Graph_explainer(self,query):
        print(f" Trend Graph of {self.file_name}")
        found_path = self.llama_inference.file_path_extractor(self.json_file , query)
        self.img_to_text.display_image(found_path)
        response = self.img_to_text.Trend_Graph(found_path, query)
        print(response)
        
    def Pairplot_explainer(self,query):
        print(f"Showing the Pairplot of {self.file_name}")
        found_path = self.llama_inference.file_path_extractor(self.json_file , query)
        self.img_to_text.display_image(found_path)
        response = self.img_to_text.PairWise_Graph(found_path, query)
        print(response)
        
    def CrossTabulation_explainer(self, query):
        var_found = self.llama_inference.varaible_selector(self.data, query)
        print(f"Showing the Cross Tabulation for varaibles {var_found[0]} and {var_found[1]}")
        found_path = self.llama_inference.file_path_extractor(self.json_file , query)      
        if os.path.isfile(f"{found_path}_{var_found[0]}_vs_{var_found[1]}.png"):
              full_path = f"{found_path}_{var_found[0]}_vs_{var_found[1]}.png"
        elif os.path.isfile(f"{found_path}_{var_found[1]}_vs_{var_found[0]}.png"):
              full_path = f"{found_path}_{var_found[1]}_vs_{var_found[0]}.png"
        else:
              full_path = None

        print("File found for your query : ", full_path)
        self.img_to_text.display_image(full_path)
        response = self.img_to_text.CrossTabulation(full_path,query,var_found[0], var_found[1])
        print(response)
        
        
    def inference_regression(self):
            try:
                file_name = self.file_name
                problem_type = self.problem_type
                source = self.source
    #             file_name = input("Enter the name of the file you want to inference from: ")
    #             if not file_name:
    #                 raise ValueError("File name cannot be empty.")

    #             problem_type = input("Enter the problem type: ")
    #             if not problem_type:
    #                 raise ValueError("Problem type cannot be empty.")

    #             source = input("Enter the source data: ")
    #             if not source:
    #                 raise ValueError("Source data cannot be empty.")

                target_variable = input("Enter the target variable: ")
                if not target_variable:
                    raise ValueError("Target variable cannot be empty.")

                input_data_str = input("Enter the input for prediction as a list of lists (e.g., {column: value, column: value} ):")
                input_data = eval(input_data_str)
                if not isinstance(input_data, dict):
                    raise ValueError("Input data should be dictionary.")

                file_path = f"Knowledge/{problem_type}/{source}/{file_name}/dataset/{file_name}"
                print("File path:", file_path)

                df = pd.read_csv(file_path)
                input_data_df = pd.DataFrame(input_data)

                inf = Inference_Regression(df, problem_type, source, file_name, target_variable)
                results = inf.regression_model(input_data_df)
                print(results)
                response = self.llama_inference.inference_regression(df, input_data, target_variable, results)
                print(response)
    #             print(f"The value of {target_variable} is: {results[0]}")

            except FileNotFoundError:
                print("File not found. Please ensure the file path is correct.")
            except ValueError as ve:
                print("Error:", ve)
            except Exception as e:
                print("An unexpected error occurred:", e)
  
    def inference_classification(self):
        try:
            file_name = self.file_name
            problem_type = self.problem_type
            source = self.source
#             file_name = input("Enter the name of the file you want to inference from: ")
#             if not file_name:
#                 raise ValueError("File name cannot be empty.")

#             problem_type = input("Enter the problem type: ")
#             if not problem_type:
#                 raise ValueError("Problem type cannot be empty.")

#             source = input("Enter the source data: ")
#             if not source:
#                 raise ValueError("Source data cannot be empty.")

            target_variable = input("Enter the target variable: ")
            if not target_variable:
                raise ValueError("Target variable cannot be empty.")

            input_data_str = input("Enter the input for prediction as a list of lists  (e.g., {column: value, column: value} ): ")
            input_data = eval(input_data_str)
            if not isinstance(input_data, dict):
                raise ValueError("Input data should be a dictionary.")

            file_path = f"Knowledge/{problem_type}/{source}/{file_name}/dataset/{file_name}"
            print("File path:", file_path)

            df = pd.read_csv(file_path)
            input_data_df = pd.DataFrame(input_data)

            inf = Inference_Classification(df, problem_type, source, file_name, target_variable)
            results = inf.classification_model(input_data_df)
            
            response = self.llama_inference.inference_classification(df, input_data, target_variable, results)
            print(response)

        except FileNotFoundError:
            print("File not found. Please ensure the file path is correct.")
        except ValueError as ve:
            print("Error:", ve)
        except Exception as e:
            print("An unexpected error occurred:", e)

            
    def inference_timeseries(self):
        pass