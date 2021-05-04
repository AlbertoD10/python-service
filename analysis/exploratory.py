from flask import jsonify, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
plt.switch_backend('agg')


def analysis(clients):

    # Create a dataframe with the users
    df_bc = pd.DataFrame(list(clients))
    # Dropping IDS
    df_bc = df_bc.drop(['_id', 'CLIENTNUM'], axis=1)

    # img = io.BytesIO()
    list_images = []

    def help(value):
        img = io.BytesIO()

        # By attrition
        plt.figure(figsize=(8, 5))
        plot = sns.countplot(x=value, hue=df_bc.Attrition_Flag)
        for p in plot.patches:
            # print(p)
            plot.annotate(p.get_height(), (p.get_x() +
                                           p.get_width()/2, p.get_height()+50))

        plt.savefig(img, format='png')
        plt.close()

        plot_url = base64.b64encode(img.getvalue()).decode()
        return plot_url
        # list_images.append(plot_url)

    list_images.append(help(df_bc.Attrition_Flag))
    list_images.append(help(df_bc.Gender))
    list_images.append(help(df_bc.Income_Category))
    list_images.append(help(df_bc.Card_Category))
    list_images.append(help(df_bc.Dependent_count))
    list_images.append(help(df_bc.Education_Level))
    list_images.append(help(df_bc.Customer_Age))
    list_images.append(help(df_bc.Dependent_count))

    return render_template('graphic.html', imagesData={'image': list_images})
