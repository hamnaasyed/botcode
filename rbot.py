#!/usr/bin/env python
# coding: utf-8

# In[31]:


import itertools
import random
from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
import matplotlib.pyplot as plt

def recommender_bot():
    decision_tree = {
        'anxiety': {
            'rarely': {
                'fear_of_worst_happening': {
                    'yes': {
                        'heart_pounding': {
                            'yes': {
                                'afraid_or_terrified': {
                                    'yes': 'You have low anxiety, recommended Course 1',
                                    'no': 'You still have low anxiety, recommended Course 1'
                                }
                            },
                            'no': {
                                'afraid_or_terrified': {
                                    'yes': 'You have kind of low anxiety, recommended Course 1',
                                    'no': 'You have Generalized Anxiety Disorder. What are you afraid of currently? Try writing down whatever is bothering you, or if you have someone to talk to, trying talking to them.'
                                }   
                            }
                        }
                    },
                    'no': {
                        'heart_pounding': {
                            'yes': {
                                'afraid_or_terrified': {
                                    'yes': 'You are panicking. Take a deep breath and click here (Course 1).',
                                    'no': 'Sit down and take a deep breath. If possible, wash your face with cold water, and go to Course 1.'
                                }
                            },
                            'no': {
                                'afraid_or_terrified': {
                                    'yes': 'What are you afraid of? If you want, you can try out these measures of Course 1 to ease your mind.',
                                    'no': 'Everything seems good with you.'
                                }   
                            }
                        }    
                    }
                },
            },
            'occasionally': {
                'fear_of_worst_happening': {
                    'yes': {
                        'heart_pounding': {
                            'yes': {
                                'difficulty_in_breathing': {
                                    'yes': {
                                        'hot_or_cold_sweats': {
                                            'yes': {
                                                'shaky_or_unsteady': {
                                                    'yes': 'You have generalized anxiety disorder. Click Course 2 to instantly cure the condition you are in.',
                                                    'no': 'You have generalized anxiety disorder. Try to calm down and click here (course 2) for further help'
                                                }
                                            },
                                            'no': {
                                                'shaky_or_unsteady': {
                                                    'yes': 'It''s normal to have anxiety from time to time. I am here for your support. Click Course 2 to have a bit of relief from your current condition',
                                                    'no': 'You have been feeling anxious lately, which is quite normal for human beings. Try to relax and visit Course 2 so you might feel a bit better.'
                                                }
                                            }   
                                        }
                                    },
                                    'no': {
                                        'hot_or_cold_sweats': {
                                            'yes': {
                                                'shaky_or_unsteady': {
                                                    'yes': 'I understand that you''re feeling anxious, and I''m here for you. Click Course 2',
                                                    'no': 'You have been feeling anxious lately, which is quite normal for human beings. Try to relax and visit Course 2 so you might feel a bit better.'
                                                }
                                            },
                                            'no': {
                                                'shaky_or_unsteady': {
                                                    'yes': 'Your feelings are valid and important. It is important to take care of yourself when you are feeling anxious so visit Course 2.',
                                                    'no': 'Its okay to feel anxious, recommended Course 1'
                                                }
                                            }   
                                        }
                                    }
                                }
                            },
                            'no': {
                                'difficulty_in_breathing': {
                                    'yes': {
                                        'hot_or_cold_sweats': {
                                            'yes': {
                                                'shaky_or_unsteady': {
                                                    'yes': 'Take some deep breaths. You are strong and view Course 2 for further help.',
                                                    'no': 'I can help you feel calm. Click Course 2.'
                                                }
                                            },
                                            'no': {
                                                'shaky_or_unsteady': {
                                                    'yes': 'Lets do something together to instantly cure the symptoms. View Course 2.',
                                                    'no': 'Take some deep breaths with me. Lets inhale for four counts, hold for four counts, and exhale for four counts. View Course 2 for further help.'
                                                }
                                            }   
                                        }
                                    },
                                    'no': {
                                        'hot_or_cold_sweats': {
                                            'yes': {
                                                'shaky_or_unsteady': {
                                                    'yes': 'It is important to take care of yourself.Make sure you are getting enough rest,exercise and healthy food. View more at Course 2.',
                                                    'no': 'It is okay to feel anxious. Your feelings are valid. Get help using Cousre 2.'
                                                }
                                            },
                                            'no': {
                                                'shaky_or_unsteady': {
                                                    'yes': 'You have generalized anxiety disorder. Click Course 2 to instantly cure the condition you are in.',
                                                    'no': 'You have Generalized Anxiety Disorder. What are you afraid of currently? Try writing down whatever is bothering you, or if you have someone to talk to, trying talking to them.'
                                                }
                                            }   
                                        }
                                    }
                                }
                            }
                        }
                    },
                    'no': {  
                        'heart_pounding': {
                            'yes': {
                                'difficulty_in_breathing': {
                                    'yes': {
                                        'hot_or_cold_sweats': {
                                            'yes': {
                                                'shaky_or_unsteady': {
                                                    'yes': 'It''s normal to have anxiety from time to time. I am here for your support. Click Course 2 to have a bit of relief from your current condition',
                                                    'no':'Take some deep breaths. I am here for your support. Click Cousre 2.'
                                                }
                                            },
                                            'no': {
                                                'shaky_or_unsteady': {
                                                    'yes': 'Try to calm down yourself. You are strong and capable. Click Course 2.',
                                                    'no': 'Sit down and take deep breath. If possible try to wash your face with cold water,View Course 2.'
                                                }
                                            }   
                                        }
                                    },
                                    'no': {
                                        'hot_or_cold_sweats': {
                                            'yes': {
                                                'shaky_or_unsteady': {
                                                    'yes': 'Sit down. Try to calm down. View Course 2 for help.',
                                                    'no': 'Take some deep breaths. I am here for your support. Click Cousre 2'
                                                }
                                            },
                                            'no': {
                                                'shaky_or_unsteady': {
                                                    'yes': 'Sit down and take deep breath. If possible try to wash your face with cold water,View Course 2',
                                                    'no': 'It is okay to feel anxious. Your feelings are valid. Get help using Cousre 2'
                                                }
                                            }   
                                        }
                                    }
                                }
                            },
                            'no': {
                                'difficulty_in_breathing': {
                                    'yes': {
                                        'hot_or_cold_sweats': {
                                            'yes': {
                                                'shaky_or_unsteady': {
                                                    'yes': 'Remember that you are not alone in this. Many people experience anxiety, and there are ways to manage it. View Course 2.',
                                                    'no': 'Sit down. Try to calm down. View Course 2 for help'
                                                }
                                            },
                                            'no': {
                                                'shaky_or_unsteady': {
                                                    'yes': 'Try to calm down yourself. You are strong and capable. Click Course 2',
                                                    'no': 'Take some deep breaths. I am here for your support. Click Cousre 2'
                                                }
                                            }   
                                        }
                                    },
                                    'no': {
                                        'hot_or_cold_sweats': {
                                            'yes': {
                                                'shaky_or_unsteady': {
                                                    'yes': 'Sit down and take deep breath. If possible try to wash your face with cold water,View Course 2',
                                                    'no': 'Go out and take some fresh air.'
                                                }
                                            },
                                            'no': {
                                                'shaky_or_unsteady': {
                                                    'yes': 'If you are cold, try to warm up yourself by doing some exercises. ',
                                                    'no': 'Everything seems fine with you.'
                                                }
                                            }   
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            'frequently': {
                'fear_of_worst_happening': {
                    'yes': { #f
                        'heart_pounding': {
                            'yes': { #hp
                                'hands_trembling': {
                                    'yes': { #ht
                                        'hot_or_cold_sweats': { 
                                            'yes': { #sweats
                                                'numbness_or_tingling': {
                                                    'yes': { #numb
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Have you considered seeking professional help? There are people who can help you manage your anxiety and feel better. Click Course 3 for instant help.',
                                                            'no': 'You''re not weak for struggling with anxiety. It takes a lot of strength to keep going when things feel overwhelming. view Course 3.'
                                                        }
                                                    },
                                                    'no' : { #numb
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'It''s okay to not be okay right now. You''re not alone because I am here to help, Click here (Course 3)',
                                                            'no': 'Take some deep breaths with me, and try to focus on the present moment.View Course 3 for further help.'
                                                        }    
                                                    }
                                                }       
                                            },
                                            'no': { #hot/cold sweats
                                                'numbness_or_tingling': {
                                                    'yes': {
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Let''s come up with a plan together to manage your anxiety and find some relief. I have this naturalistic course 3 for your instant help.',
                                                            'no': 'It''s okay to feel anxious, and it''s okay to ask for help when you need it. click Course 3 for help'
                                                        }
                                                    },
                                                    'no' : {
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Take some deep breaths with me, and try to focus on the present moment.View Course 3',
                                                            'no': 'It''s normal to have anxiety from time to time. I am here for your support. Click Course 2 to have a bit of relief from your current condition.'
                                                        }    
                                                    }
                                                }       
                                            }
                                        }
                                    },
                                    'no': { #handstrembling
                                        'hot_or_cold_sweats': { 
                                            'yes': { #sweats
                                                'numbness_or_tingling': {
                                                    'yes': { #numb
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'It''s okay to feel anxious, and it''s okay to ask for help when you need it. click Course 3 for help',
                                                            'no': 'Take some deep breaths. I am here for your support. Click Cousre 2'
                                                        }
                                                    },
                                                    'no' : { #numb
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'It''s normal to have anxiety from time to time. I am here for your support. Click Course 2 to have a bit of relief from your current condition',
                                                            'no': 'It is okay to feel anxious. Your feelings are valid. Get help using Cousre 2'
                                                        }    
                                                    }
                                                }       
                                            },
                                            'no': { #hot/cold sweats
                                                'numbness_or_tingling': {
                                                    'yes': {
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Have you considered seeking professional help? There are people who can help you manage your anxiety and feel better',
                                                            'no': 'Lets come up with a plan together to manage your anxiety and find some relief. I have this naturalistic course 3 for your instant help.'
                                                        }
                                                    },
                                                    'no' : {
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Its okay to feel anxious, and its okay to ask for help when you need it. click Course 3 for help',
                                                            'no': 'Remember that you are not alone in this. Many people experience anxiety, and there are ways to manage it. View Course 2'
                                                        }    
                                                    }
                                                }       
                                            }
                                        }   
                                    }
                                }
                            },
                            'no' : { #heart_pounding
                                'hands_trembling': {
                                    'yes': { #ht
                                        'hot_or_cold_sweats': { 
                                            'yes': { #sweats
                                                'numbness_or_tingling': {
                                                    'yes': { #numb
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'You are not weak for struggling with anxiety. It takes a lot of strength to keep going when things feel overwhelming. view Course 3',
                                                            'no': 'Its okay to feel anxious, and its okay to ask for help when you need it. click Course 3 for help'
                                                        }
                                                    },
                                                    'no' : { #numb
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Its okay to not be okay right now. You are not alone because I am here to help, Click here (Course 3)',
                                                            'no': 'Sit down. Try to calm down. View Course 2 for help'
                                                        }    
                                                    }
                                                }       
                                            },
                                            'no': { #hot/cold sweats
                                                'numbness_or_tingling': {
                                                    'yes': {
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Have you considered seeking professional help? There are people who can help you manage your anxiety and feel better. View Course 3.',
                                                            'no': 'Its okay to feel anxious, and its okay to ask for help when you need it. click Course 3 for help'
                                                        }
                                                    },
                                                    'no' : {
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Its okay to not be okay right now. You are not alone because I am here to help, Click here (Course 3)',
                                                            'no': 'Try to calm down yourself. You are strong and capable. Click Course 2.'
                                                        }    
                                                    }
                                                }       
                                            }
                                        }   
                                    },
                                    'no': { #handstrembling
                                        'hot_or_cold_sweats': { 
                                            'yes': { #sweats
                                                'numbness_or_tingling': {
                                                    'yes': { #numb
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'You are not weak for struggling with anxiety. It takes a lot of strength to keep going when things feel overwhelming. view Course 3',
                                                            'no': 'You have been feeling anxious lately, which is quite normal for human beings. Try to relax and visit Course 2 so you might feel a bit better.'
                                                        }
                                                    },
                                                    'no' : { #numb
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Sit down. Try to calm down. View Course 2 for help',
                                                            'no': 'Try to calm down. View Course 2 for help'
                                                        }    
                                                    }
                                                }       
                                            },
                                            'no': { #hot/cold sweats
                                                'numbness_or_tingling': {
                                                    'yes': {
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Its okay to feel anxious, and its okay to ask for help when you need it. click Course 3 for help',
                                                            'no': 'Its okay to not be okay right now. You are not alone because I am here to help, Click here (Course 3)'
                                                        }
                                                    },
                                                    'no' : {
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Sit down. Try to calm down. View Course 2 for help',
                                                            'no': 'You have low anxiety, recommended Course 1'
                                                        }    
                                                    }
                                                }       
                                            }
                                        }   
                                    }
                                }
                            }
                        }
                    },
                    'no' : { #fear_of_worst_happening
                        'heart_pounding': {
                            'yes': { #hp
                                'hands_trembling': {
                                    'yes': { #ht
                                        'hot_or_cold_sweats': { 
                                            'yes': { #sweats
                                                'numbness_or_tingling': {
                                                    'yes': { #numb
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Lets come up with a plan together to manage your anxiety and find some relief. I have this naturalistic course 3 for your instant help.',
                                                            'no': 'You are not weak for struggling with anxiety. It takes a lot of strength to keep going when things feel overwhelming. view Course 3'
                                                        }
                                                    },
                                                    'no' : { #numb
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Take some deep breaths with me, and try to focus on the present moment.View Cousrse 3',
                                                            'no': 'Sit down. Try to calm down. View Course 2 for help'
                                                        }    
                                                    }
                                                }       
                                            },
                                            'no': { #hot/cold sweats
                                                'numbness_or_tingling': {
                                                    'yes': {
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'You are not weak for struggling with anxiety. It takes a lot of strength to keep going when things feel overwhelming. view Course 3',
                                                            'no': 'Its okay to not be okay right now. You are not alone because I am here to help, Click here (Course 3)'
                                                        }
                                                    },
                                                    'no' : {
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Have you considered seeking professional help? There are people who can help you manage your anxiety and feel better. View Course 3',
                                                            'no': 'Sit down. Try to calm down. View Course 2 for help'
                                                        }    
                                                    }
                                                }       
                                            }
                                        }   
                                    },
                                    'no': { #handstrembling
                                        'hot_or_cold_sweats': { 
                                            'yes': { #sweats
                                                'numbness_or_tingling': {
                                                    'yes': { #numb
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Lets come up with a plan together to manage your anxiety and find some relief. I have this naturalistic course 3 for your instant help.',
                                                            'no': 'Its okay to feel anxious, and its okay to ask for help when you need it. click Course 3 for help'
                                                        }
                                                    },
                                                    'no' : { #numb
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Sit down. Try to calm down. View Course 2 for help',
                                                            'no': 'Try to calm down yourself. You are strong and capable. Click Course 2'
                                                        }    
                                                    }
                                                }       
                                            },
                                            'no': { #hot/cold sweats
                                                'numbness_or_tingling': {
                                                    'yes': {
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Its okay to not be okay right now. You are not alone because I am here to help, Click here (Course 3)',
                                                            'no': 'Take some deep breaths with me, and try to focus on the present moment.View Course 3'
                                                        }
                                                    },
                                                    'no' : {
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Try to calm down yourself. You are strong and capable. Click Course 2',
                                                            'no': 'Sit down. Try to calm down. View Course 1 for help'
                                                        }    
                                                    }
                                                }       
                                            }
                                        }   
                                    }
                                }
                            },
                            'no' : { #heart_pounding
                                'hands_trembling': {
                                    'yes': { #ht
                                        'hot_or_cold_sweats': { 
                                            'yes': { #sweats
                                                'numbness_or_tingling': {
                                                    'yes': { #numb
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Have you considered seeking professional help? There are people who can help you manage your anxiety and feel better. View Course 3',
                                                            'no': 'Lets come up with a plan together to manage your anxiety and find some relief. I have this naturalistic course 3 for your instant help.'
                                                        }
                                                    },
                                                    'no' : { #numb
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Take some deep breaths with me, and try to focus on the present moment.View Course 3',
                                                            'no': 'Try to calm down yourself. You are strong and capable. Click Course 2'
                                                        }    
                                                    }
                                                }       
                                            },
                                            'no': { #hot/cold sweats
                                                'numbness_or_tingling': {
                                                    'yes': {
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Have you considered seeking professional help? There are people who can help you manage your anxiety and feel better. View Course 3.',
                                                            'no': 'Its okay to not be okay right now. You are not alone because I am here to help, Click here (Course 3)'
                                                        }
                                                    },
                                                    'no' : {
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Lets come up with a plan together to manage your anxiety and find some relief. I have this naturalistic course 3 for your instant help.',
                                                            'no': 'Try to calm down yourself. You are strong and capable. Click Course 2'
                                                        }    
                                                    }
                                                }       
                                            }
                                        }   
                                    },
                                    'no': { #handstrembling
                                        'hot_or_cold_sweats': { 
                                            'yes': { #sweats
                                                'numbness_or_tingling': {
                                                    'yes': { #numb
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'You are not weak for struggling with anxiety. It takes a lot of strength to keep going when things feel overwhelming. view Course 3',
                                                            'no': 'Take some deep breaths with me, and try to focus on the present moment. View Course 3'
                                                        }
                                                    },
                                                    'no' : { #numb
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Take some deep breaths. I am here for your support. Click Cousre 2',
                                                            'no': 'Try to calm down yourself. You are strong and capable. Click Course 2'
                                                        }    
                                                    }
                                                }       
                                            },
                                            'no': { #hot/cold sweats
                                                'numbness_or_tingling': {
                                                    'yes': {
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Do some movement and view Course 3.',
                                                            'no': 'Do some movement and view Course 3'
                                                        }
                                                    },
                                                    'no' : {
                                                        'wobbliness_in_legs' : {
                                                            'yes' : 'Try to eat something and view Course 3 for further help.',
                                                            'no': 'Everthing seems fine with you.'
                                                        }    
                                                    }
                                                }       
                                            }
                                        }   
                                    }
                                }
                            }
                        }
                    }
                },
            }
        }
    }

def course_selector():
    print("Hi there! I'm a Seggie. I can help you cure your anxiety instantly based on your symptoms Kindly cooperate.")
    anxiety = input("Feeling anxious lately? (rarely, occasionally, frequently)")
    
    if anxiety == 'rarely':
        fear_of_worst_happening = input("Are you feeling that something worse will happen? (yes, no) ")
        heart_pounding = input("Is your heartbeat fast? (yes or no) ")
        afraid_or_terrified = input("Are you terrified of something right now? (yes or no) ")  
        anxiety_level = decision_tree['anxiety'][anxiety]['fear_of_worst_happening'][fear_of_worst_happening]['heart_pounding'][heart_pounding]['afraid_or_terrified'][afraid_or_terrified]
    
    elif anxiety == 'occasionally':
        fear_of_worst_happening = input("Are you feeling that something worse will happen? (yes, no) ")
        heart_pounding = input("Is your heartbeat fast? (yes or no) ")
        difficulty_in_breathing=input("Are you have difficulting in breathing? (yes,no)")
        hot_or_cold_sweats = input("Are you sweating right now? hot/cold sweats? (yes, no)")
        shaky_or_unsteady = input("Are you trembling? feeling unstable or shaky? (yes no)")
        anxiety_level = decision_tree['anxiety'][anxiety]['fear_of_worst_happening'][fear_of_worst_happening]['heart_pounding'][heart_pounding]['difficulty_in_breathing'][difficulty_in_breathing]['hot_or_cold_sweats'][hot_or_cold_sweats]['shaky_or_unsteady'][shaky_or_unsteady]
    
    elif anxiety == 'frequently':
        fear_of_worst_happening = input("Are you feeling that something worse will happen? (yes, no) ")
        heart_pounding = input("Is your heartbeat fast? (yes or no) ")
        hands_trembling=input("Do you feel like your hands are trembling or shaking? (yes,no)")
        hot_or_cold_sweats = input("Are you sweating right now? hot/cold sweats? (yes, no)")
        numbness_or_tingling = input("Do you feel like your body is numb? or do you have a tingling sensation? Do you feel any pins and needles sensation in your body when you feel anxious? (yes, no)")
        wobbliness_in_legs = input("Do you feel like your legs are shaky or wobbly when you're anxious? (yes, no)")
        anxiety_level = decision_tree['anxiety'][anxiety]['fear_of_worst_happening'][fear_of_worst_happening]['heart_pounding'][heart_pounding]['hands_trembling'][hands_trembling]['hot_or_cold_sweats'][hot_or_cold_sweats]['numbness_or_tingling'][numbness_or_tingling]['wobbliness_in_legs'][wobbliness_in_legs]    
        
    print(f"Based on your answers, {anxiety_level}.")
 
def bot_listener():
    more_info = input("Would you like to talk? (yes or no) ")
    if more_info == 'yes':
        print("Great! What kind of issue are you currently facing?")
        print("Here are some common categories: Relationship, Financial, Career, Health, Life problems")
        problem = input("Please choose a category: ")
        if problem.lower() == 'relationship':
            input("I'm sorry to hear that. Please tell your problem." )
            print("It sounds like you're going through a tough time in your relationship. Remember, communication is key in any relationship. Have you tried talking to your partner about how you feel?")
            print("If you need more support, there are many articles available on our website, you can get help from there.")

        elif problem.lower() == 'financial':
            input("I'm sorry to hear that you're experiencing financial difficulties. What is the matter?")
            print("This can be a very stressful situation. It's important to take a step back and assess your financial situation.")
            print("You can find help from our website regarding budgets and how you can cut down on your expenses.")

        elif problem.lower() == 'career':
            input("It sounds like you're facing some challenges in your career. This is absolutely normal. What are you going through?")
            print("Remember, it's normal to face setbacks and challenges in any career.")
            print("Have you considered talking to a mentor or career counselor for support and guidance? You can contact the expert help from the website and you can also refer to the articles that guide you in this field.")

        elif problem.lower() == 'health':
            input("What are your health concerns? Are you okay?")
            print("I'm sorry to hear that you're experiencing health issues. This can be a difficult and scary situation. It's important to prioritize your health and seek medical advice if necessary.")
            print("Remember, taking care of your physical and mental health is crucial for overall wellbeing.")

        elif problem.lower() == 'life problems':
            input("Everything is going to be okay. Have Faith in yourself and God. What is the matter?")
            print("It sounds like you're going through some tough times in your life. Remember, it's okay to not be okay.")
            print("If you need support, there are many resources available such as therapy, support groups, and hotlines.")
        
        else:
            print("I'm sorry, I don't understand. Please try again.")

        rant = input("Would you like to share more about what's been bothering you? (yes or no) ")
        if rant == 'yes':
            input("I'm here to listen. Please feel free to share your thoughts and feelings.")
            print("Remember, sometimes just talking about your problems can help. I am a bot therefore I do not have expertise in problems regarding human beings, though you can always seek guidance from the help provided in our website")
        else:
            print("Okay, let me know if you need anything else. Take care!")

    else:
        print("Okay, let me know if you need anything else. Take care!")

#Call the chatbot function
course_selector()
bot_listener()

if __name__ == '__main__':
    recommender_bot()


# In[ ]:




