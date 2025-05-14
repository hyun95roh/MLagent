import numpy as np
import plotly.graph_objects as go

class gaussian():
    def __init__(self) -> None:
        # Instance Attributes 
        self.x = np.linspace(-5, 5, 100) 
        self.y = np.linspace(-5, 5, 100)
        self.coef_pi, self.coef_exp = 0.5, 0.5
        self.coef_x, self.coef_y = 1, 1 
        self.xpower, self.ypower = 2, 2 
        self.X, self.Y = np.meshgrid(self.x, self.y) # Create a grid of x and y values
        self.exp_bias, self.Z = 0, None  
        pass
    

    def generate_data(self):
        """
        Generates the data for the 3D surface plot.
        """
        # Calculate the PDF at each point
        self.Z = self.coef_pi/(np.pi) * np.exp(-(self.coef_x*self.X**self.xpower + self.coef_y*self.Y**self.ypower) * self.coef_exp + self.exp_bias) 


    def create_plot(self,title):
        """
        Creates the 3D surface plot using Plotly.
        """
        fig = go.Figure(data=[go.Surface(x=self.X, y=self.Y, z=self.Z, colorscale='Viridis', opacity=0.8)])

        # Set the title and axis labels
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='PDF'
            )
        )
        return fig