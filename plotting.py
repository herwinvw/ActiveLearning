from plotnine import *
import numpy as np
import pandas

def plot_sample(sample, data):
    return ggplot(sample, aes(x='x', y='y', fill='label', shape='label'))+\
            geom_point(size=3)+\
            geom_point(data, aes(x='x',y='y',color='label'))

def plot_linear_classifier(sample, data, lg):
    w0 = lg.intercept_[0]
    w1, w2 = lg.coef_[0]
    def boundary(x):
        return (-w0/w2)+(-w1/w2)*x
    return (ggplot(sample, aes(x='x', y='y', fill='label', shape='label'))+\
        geom_point(size=3)+geom_segment(x=-5,xend=5,y=boundary(-5),yend=boundary(5))+\
        geom_point(data, aes(x='x',y='y',color='label')))

def plot_classifier(sample, data, cf):
    x_min, x_max = data.x.min(), data.x.max()
    y_min, y_max = data.y.min(), data.y.max()
    plot_step = 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
    xy = np.c_[xx.ravel(), yy.ravel()]
    z = cf.predict(xy)
    df_plot = pandas.DataFrame({'x':xy[:,0],'y':xy[:,1],'label':z})
    return (ggplot(sample, aes(x='x', y='y', fill='label', shape='label'))+\
        geom_point(size=3)+\
        geom_point(data, aes(x='x',y='y',color='label'))+\
        geom_tile(df_plot, alpha=0.5))