# OcularDrugToxicityPrediction_ResearchPractice
Abstract :
There are many systemic drugs which can cause ocular toxicity due to their accumulation at non-target site. Hydroxychloroquine is one of the example of a drug causing induced ocular toxicity. It is caused when the drugs gain access to the eyes through the transporters. So, structural fingerprints within the drug molecules will have to be recognized as substrate/non-substrate against a particular transporter. To protect people from potential eye injury caused due to these toxic drugs, several in-lab experiments have to be carried out including animal screening, which is both time-consuming and costly. To assist the pharmacological testing, several computational models can be made to tag potential chemical compounds as substrates or non-substrates and then testing only the high risk molecules (probably the substrates) for further research. This virtual screening using machine learning models is thus quite efficient and attractive to the toxicologists for tagging potential ocular toxicants. Various Supervised Machine Learning models have been used to identify whether the molecule is a substrate or non-substrate, so that the toxicity of the drugs can be predicted.


Folder structure :
1. DataSheets - has input datasheet that is used to train the model
2. KfoldOutputSheet - has the output sheet created by K-fold cross validation
3. OutputSheet - has the excel sheets created at the end to store the prediction of testing molecules
4. Plots - contains images of all the plots created via code
5. TrainSheetForLogReg - stores the data which is taken as input for Logistic Regression model
6. outputs/rules - stores the decision tree created for Decision Tree and C4.5 model


File details :
1. Final_merged_kfold.py --> has main function and has the final code that needs to be executed.
2. Final report_RP.pdf --> is the Research practice final report (pdf format)
3. RP_MidSemsReport_DrugToxicityPrediction.pdf --> is the report submitted for midsemester exams
4. SupervisedLearningodel_RandomSplit.py --> has the model implementation without k-fold cross validation, using random splitting of dataset.
5. graph_plots.py --> is a utility file which has functions to create all the plots
6. modles.py --> has the model implementation functions
7. utilities.py --> has utility functions to scale, handle missing values, generate dataframe, create excel etc.
