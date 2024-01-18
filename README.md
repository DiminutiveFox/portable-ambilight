# Macro Creator 
It increases productivity and simplifies time-consuming and repetitive tasks. 
Personally it helped me during SCADA development process (visualisation of a factory) 

# Project description
Project contains 2 main areas - Macro Generator and Object Finder. Macro Generator is meant for creating and running macros, when Object Finder looks for a determined object on the screen and finds it's coordinates - then macro can be performed upon it. 

![image](https://github.com/DiminutiveFox/MacroCreator/assets/135659343/adbb901c-f3fe-4dee-b2bf-08ed890bc7ce)

# Macro Generator
It allows user to create macros and execute them in multiple ways. 

'Generate Macro' button runs a sub-process which main purpose is to register user's mouse and keyboard input and store it in a .csv file in a 'macros' directory. 
In drop-down menus, user can pick up to 3 different macros that will be executed in a single run. Macro files can also be opened and edited - files are structured in an intuitive way. There are three exceptions, that user must be aware of:
- esc button ends listening and saves the macro in a file specified in the entry area. Esc button is not saved and cannot be used in a macro sequence.
- when '$' appears in a 'Key' column, the sub-procces tries to type the text written in the 'X' column. It is useful when there is a need to type a specified text during the sequence, however user needs to specify it directly in the file
- when '#' appears in a 'Key' column, the sub-process tries to type the current value of the counter. It is useful when there is a need to eg. change the filenames and add an identificator at the end of every name. It is only true, when 'Counter' checkbox is checked. In 'Object Finder' and 'Listen and Run' modes counter is incremented at the end of every macros' cycle. User can specify the start value and the addend when checkbox is clicked.

![image](https://github.com/DiminutiveFox/MacroCreator/assets/135659343/7b19f85c-74a5-46e2-9283-0d9f2839aacd)

Trigger button is used by 'Listen and Run'. Here user can specify the button that will run the macro during that process. By default the mouse's scroll button is selected.
User can select 'Working with offset' function that calculates mouse movements based on a loaded macro and current mouse position so that mouse will run with an 'offset'.
User can specify the number of repeats - after macro execution it will be repeated as many times as it is specified in the drop-down menu. 

Finally macros' can be run using two buttons - 'Run' and 'Listen and Run'. First button is dedicated for tests - it minimizes the window and runs macro once and then repeats it, if 'Repeatable' box is clicked. 'Listen and Run' 
process listens for user's input and runs macro, when button specified in 'Trigger Button' area is detected. It will not stop to listen unless you press esc button.

# Object Finder
It allows user to find object on the screen and perform macro on every of these objects. 

Firstly, screenshots of objects have to be placed in the 'images' directory. It has to be done manually (Windows eg. pressing 'Shift + Win + S'). 
Filename has to end with a .csv extension - otherwise the coordinates won't be saved. User needs to determine the accuracy coefficient - it has to be between 0 and 0.99 - the lower the value, the more positive falses 
will be found; the higher the value, the more false positives.

![image](https://github.com/DiminutiveFox/MacroCreator/assets/135659343/f14a594d-01b5-4c19-ba2f-0a8206e85996)

Files are stored in a 'locations' directory. Application also creates and shows a picture with object that were found - objects are indicated by green squares. User can validate and play with accuracy coefficient to improve the results. However the file cannot be picked in the drop-down menu unless the 'Working with object finder' checkbox is clicked.

Start button is enabled when correct file is picked in drop-down menu. All checkboxes from 'Macro Generator' area are applied here. 
