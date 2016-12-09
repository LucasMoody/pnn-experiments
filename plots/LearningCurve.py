import plotly
import plotly.graph_objs as go

def plotLearningCurve(metrics):
    # Create a trace
    data = map(lambda (values, ranges, name): go.Scatter(y = values, x = ranges, name=name), metrics)
    plotly.offline.plot({
        "data": data,
        "layout": go.Layout(title="hello world")
    }, filename="reporting.html")

