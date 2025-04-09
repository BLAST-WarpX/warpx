# script-version: 2.0
# Catalyst state generated using paraview version 5.11.1-1332-ga0a402a54e
# and validated with 5.12
import paraview

paraview.compatibility.major = 5
paraview.compatibility.minor = 12

#### import the simple module from the paraview
from paraview.simple import *  # noqa: F403

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# Create a new 'Render View'
renderView1 = CreateView("RenderView")
renderView1.ViewSize = [889, 820]
renderView1.AxesGrid = "Grid Axes 3D Actor"
renderView1.OrientationAxesVisibility = 0
renderView1.StereoType = "Crystal Eyes"
renderView1.CameraPosition = [
    0.537460346333711,
    -0.7885714264869157,
    0.47540803206975096,
]
renderView1.CameraFocalPoint = [
    -0.002360633658114411,
    -0.0033755520058871905,
    -0.0028772574955099155,
]
renderView1.CameraViewUp = [
    -0.010196376455648859,
    0.5150674559093057,
    0.8570889975786005,
]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 0.8660254037844386
renderView1.LegendGrid = "Legend Grid Actor"
renderView1.UseColorPaletteForBackground = 0
renderView1.Background = [0.08627450980392157, 0.06274509803921569, 0.11372549019607843]

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name="Layout #1")
layout1.AssignView(0, renderView1)
layout1.SetSize(889, 820)

# ----------------------------------------------------------------
# restore active view
# ----------------------------------------------------------------
SetActiveView(renderView1)

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML Partitioned Dataset Collection Reader'
mesh = PVTrivialProducer(registrationName="mesh")

# create a new 'Calculator'
calculator1 = Calculator(registrationName="Calculator1", Input=mesh)
calculator1.AttributeType = "Cell Data"
calculator1.ResultArrayName = "E"
calculator1.Function = "sqrt(Ex^2 + Ey^2 + Ez^2)"

# create a new 'Threshold'
threshold1 = Threshold(registrationName="Threshold1", Input=calculator1)
threshold1.Scalars = ["CELLS", "E"]
threshold1.LowerThreshold = 0.00014553574512481541
threshold1.UpperThreshold = 0.00033841915322276836

# create a new 'Clip'
clip1 = Clip(registrationName="Clip1", Input=threshold1)
clip1.ClipType = "Plane"
clip1.HyperTreeGridClipper = "Plane"
clip1.Scalars = ["CELLS", "E"]
clip1.Value = 0.00034051798664982454

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from clip1
clip1Display = Show(clip1, renderView1, "UnstructuredGridRepresentation")

# get 2D transfer function for 'E'
eTF2D = GetTransferFunction2D("E")
eTF2D.ScalarRangeInitialized = 1
eTF2D.Range = [2.7209191271959258e-08, 0.00033841915322276836, 0.0, 1.0]

# get color transfer function/color map for 'E'
eLUT = GetColorTransferFunction("E")
eLUT.TransferFunction2D = eTF2D
eLUT.RGBPoints = [
    0.00014565122778316538,
    0.23137254902,
    0.298039215686,
    0.752941176471,
    0.0002596781039235302,
    0.865,
    0.865,
    0.865,
    0.00037370498006389504,
    0.705882352941,
    0.0156862745098,
    0.149019607843,
]
eLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'E'
ePWF = GetOpacityTransferFunction("E")
ePWF.Points = [
    0.00014565122778316538,
    0.0,
    0.5,
    0.0,
    0.00037370498006389504,
    1.0,
    0.5,
    0.0,
]
ePWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display.Representation = "Surface"
clip1Display.ColorArrayName = ["CELLS", "E"]
clip1Display.LookupTable = eLUT
clip1Display.SelectTCoordArray = "None"
clip1Display.SelectNormalArray = "None"
clip1Display.SelectTangentArray = "None"
clip1Display.OSPRayScaleFunction = "Piecewise Function"
clip1Display.Assembly = "Hierarchy"
clip1Display.SelectOrientationVectors = "None"
clip1Display.ScaleFactor = 0.078125
clip1Display.SelectScaleArray = "E"
clip1Display.GlyphType = "Arrow"
clip1Display.GlyphTableIndexArray = "E"
clip1Display.GaussianRadius = 0.00390625
clip1Display.SetScaleArray = [None, ""]
clip1Display.ScaleTransferFunction = "Piecewise Function"
clip1Display.OpacityArray = [None, ""]
clip1Display.OpacityTransferFunction = "Piecewise Function"
clip1Display.DataAxesGrid = "Grid Axes Representation"
clip1Display.PolarAxes = "Polar Axes Representation"
clip1Display.ScalarOpacityFunction = ePWF
clip1Display.ScalarOpacityUnitDistance = 0.03870829665815266
clip1Display.OpacityArrayName = ["CELLS", "E"]
clip1Display.SelectInputVectors = [None, ""]
clip1Display.WriteLog = ""

# setup the color legend parameters for each legend in this view

# get color legend/bar for eLUT in view renderView1
eLUTColorBar = GetScalarBar(eLUT, renderView1)
eLUTColorBar.WindowLocation = "Any Location"
eLUTColorBar.Position = [0.7840269966254219, 0.3353658536585365]
eLUTColorBar.Title = "E"
eLUTColorBar.ComponentTitle = ""
eLUTColorBar.ScalarBarLength = 0.3300000000000001

# set color bar visibility
eLUTColorBar.Visibility = 1

# show color legend
clip1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity maps used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup animation scene, tracks and keyframes
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get time animation track
timeAnimationCue1 = GetTimeTrack()

# initialize the animation scene

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# initialize the timekeeper

# initialize the animation track

# get animation scene
animationScene1 = GetAnimationScene()

# initialize the animation scene
animationScene1.ViewModules = renderView1
animationScene1.Cues = timeAnimationCue1
animationScene1.AnimationTime = 3.000000000000001e-05
animationScene1.EndTime = 3.000000000000001e-05
animationScene1.PlayMode = "Snap To TimeSteps"

# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------

# create extractor
pNG1 = CreateExtractor("PNG", renderView1, registrationName="PNG1")
# trace defaults for the extractor.
pNG1.Trigger = "Time Step"

# init the 'PNG' selected for 'Writer'
pNG1.Writer.FileName = "RenderView1_{timestep:06d}{camera}.png"
pNG1.Writer.ImageResolution = [889, 820]
pNG1.Writer.Format = "PNG"

# ----------------------------------------------------------------
# restore active source
# ----------------------------------------------------------------
SetActiveSource(threshold1)

# ------------------------------------------------------------------------------
# Catalyst options
# ------------------------------------------------------------------------------
from paraview import catalyst

options = catalyst.Options()
options.GlobalTrigger = "Time Step"
options.CatalystLiveTrigger = "Time Step"

if __name__ == "__main__":
    from paraview.simple import SaveExtractsUsingCatalystOptions

    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)
