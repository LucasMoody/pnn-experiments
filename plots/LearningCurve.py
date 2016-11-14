import plotly
import plotly.graph_objs as go

def plotLearningCurve(metrics):
    # Create a trace
    data = map(lambda (values, name): go.Scatter(y = values, name=name), metrics)
    plotly.offline.plot({
        "data": data,
        "layout": go.Layout(title="hello world")
    })

