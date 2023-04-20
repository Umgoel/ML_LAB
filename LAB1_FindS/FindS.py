{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8ee316af",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "9b39036a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "56856607",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_csv('FindS.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "1a78d94f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sky</th>\n",
       "      <th>airTemp</th>\n",
       "      <th>humidity</th>\n",
       "      <th>wind</th>\n",
       "      <th>water</th>\n",
       "      <th>enjoySport</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>sunny</th>\n",
       "      <td>warm</td>\n",
       "      <td>normal</td>\n",
       "      <td>strong</td>\n",
       "      <td>warm</td>\n",
       "      <td>same</td>\n",
       "      <td>yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sunny</th>\n",
       "      <td>warm</td>\n",
       "      <td>high</td>\n",
       "      <td>strong</td>\n",
       "      <td>warm</td>\n",
       "      <td>same</td>\n",
       "      <td>yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>rainy</th>\n",
       "      <td>cold</td>\n",
       "      <td>high</td>\n",
       "      <td>strong</td>\n",
       "      <td>warm</td>\n",
       "      <td>change</td>\n",
       "      <td>no</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sunny</th>\n",
       "      <td>warm</td>\n",
       "      <td>high</td>\n",
       "      <td>strong</td>\n",
       "      <td>cool</td>\n",
       "      <td>change</td>\n",
       "      <td>yes</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        sky airTemp humidity  wind   water enjoySport\n",
       "sunny  warm  normal   strong  warm    same        yes\n",
       "sunny  warm    high   strong  warm    same        yes\n",
       "rainy  cold    high   strong  warm  change         no\n",
       "sunny  warm    high   strong  cool  change        yes"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "91069d73",
   "metadata": {},
   "outputs": [],
   "source": [
    "concepts = np.array(data)[:,:-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "32d60c30",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([['warm', 'normal', 'strong', 'warm', 'same'],\n",
       "       ['warm', 'high', 'strong', 'warm', 'same'],\n",
       "       ['cold', 'high', 'strong', 'warm', 'change'],\n",
       "       ['warm', 'high', 'strong', 'cool', 'change']], dtype=object)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "concepts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "5b2fed5c",
   "metadata": {},
   "outputs": [],
   "source": [
    "target = np.array(data)[:,-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "f707b24a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['yes', 'yes', 'no', 'yes'], dtype=object)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "0b638631",
   "metadata": {},
   "outputs": [],
   "source": [
    "def train(con,tar):\n",
    "    for i,val in enumerate(tar):\n",
    "        if val=='yes':\n",
    "            specific_h = con[i].copy()\n",
    "            break\n",
    "            \n",
    "    for i,val in enumerate(con):\n",
    "        if tar[i]=='yes':\n",
    "            for x in range(len(specific_h)):\n",
    "                if val[x] != specific_h[x]:\n",
    "                    specific_h[x] = '?'\n",
    "                else:\n",
    "                    pass\n",
    "    return specific_h"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "6669bd25",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['warm', '?', 'strong', '?', '?'], dtype=object)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train(concepts, target)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1f516cf2",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
