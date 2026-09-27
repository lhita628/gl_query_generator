# =============================================================================
# IMPORTS & ENVIRONMENT SETUP
# =============================================================================

# Standard Library
import os
import json
import re
import warnings
from typing import TypedDict, List, Any, Dict, Optional

# Data Manipulation & Database
import pandas as pd
import sqlite3
import sqlparse

# LangChain & OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# Warnings Management
warnings.filterwarnings('ignore')