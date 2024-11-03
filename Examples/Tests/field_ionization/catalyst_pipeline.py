# script-version: 2.0
# Catalyst state generated using paraview version 5.11.1-1332-ga0a402a54e
# and validated with paraview 5.12,1
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
renderView1.InteractionMode = "2D"
renderView1.AxesGrid = "Grid Axes 3D Actor"
renderView1.CenterOfRotation = [5.125578491129211e-06, 5.125219398642561e-06, 0.0]
renderView1.StereoType = "Crystal Eyes"
renderView1.CameraPosition = [
    -3.2986377960746266e-08,
    1.2519774889107966e-05,
    6.655318906751408e-05,
]
renderView1.CameraFocalPoint = [-3.2986377960746266e-08, 1.2519774889107966e-05, 0.0]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 9.246257677588838e-06
renderView1.LegendGrid = "Legend Grid Actor"
renderView1.UseColorPaletteForBackground = 0
renderView1.Background = [0.13725490196078433, 0.1450980392156863, 0.17647058823529413]

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
particles = PVTrivialProducer(registrationName="particles")

# create a new 'Clip'
clip1 = Clip(registrationName="Clip1", Input=particles)
clip1.ClipType = "Plane"
clip1.HyperTreeGridClipper = "Plane"
clip1.Scalars = ["CELLS", "particle_electrons_cpu"]
clip1.Value = -396581.0
clip1.Invert = 0

# init the 'Plane' selected for 'ClipType'
clip1.ClipType.Origin = [5.12557849112921e-06, 5.02521939864256e-06, 0.0]
clip1.ClipType.Normal = [0.0, 1.0, 0.0]

# init the 'Plane' selected for 'HyperTreeGridClipper'
clip1.HyperTreeGridClipper.Origin = [5.12557849112921e-06, 5.12521939864256e-06, 0.0]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from clip1
clip1Display = Show(clip1, renderView1, "UnstructuredGridRepresentation")

# get 2D transfer function for 'particle_electrons_ux'
particle_electrons_uxTF2D = GetTransferFunction2D("particle_electrons_ux")
particle_electrons_uxTF2D.ScalarRangeInitialized = 1
particle_electrons_uxTF2D.Range = [-719020005.8608257, 713207105.8017796, 0.0, 1.0]

# get color transfer function/color map for 'particle_electrons_ux'
particle_electrons_uxLUT = GetColorTransferFunction("particle_electrons_ux")
particle_electrons_uxLUT.TransferFunction2D = particle_electrons_uxTF2D
particle_electrons_uxLUT.RGBPoints = [
    -813777010.3574873,
    0.0862745098039216,
    0.00392156862745098,
    0.298039215686275,
    -764496926.60325,
    0.113725,
    0.0235294,
    0.45098,
    -723569500.3656244,
    0.105882,
    0.0509804,
    0.509804,
    -695170821.8676349,
    0.0392157,
    0.0392157,
    0.560784,
    -667607367.8918439,
    0.0313725,
    0.0980392,
    0.6,
    -640879233.4148993,
    0.0431373,
    0.164706,
    0.639216,
    -602457575.7205275,
    0.054902,
    0.243137,
    0.678431,
    -551506980.3332543,
    0.054902,
    0.317647,
    0.709804,
    -488862937.3087116,
    0.0509804,
    0.396078,
    0.741176,
    -448248678.177614,
    0.0392157,
    0.466667,
    0.768627,
    -407634419.0465164,
    0.0313725,
    0.537255,
    0.788235,
    -365245168.54020303,
    0.0313725,
    0.615686,
    0.811765,
    -321811973.7593288,
    0.0235294,
    0.709804,
    0.831373,
    -278378684.00181174,
    0.0509804,
    0.8,
    0.85098,
    -242462794.85069358,
    0.0705882,
    0.854902,
    0.870588,
    -209052579.2661966,
    0.262745,
    0.901961,
    0.862745,
    -179818676.24599826,
    0.423529,
    0.941176,
    0.87451,
    -134714937.4440689,
    0.572549,
    0.964706,
    0.835294,
    -104645714.92502928,
    0.658824,
    0.980392,
    0.843137,
    -82929117.53459203,
    0.764706,
    0.980392,
    0.866667,
    -59542023.61142123,
    0.827451,
    0.980392,
    0.886275,
    -13603012.798971772,
    0.913725,
    0.988235,
    0.937255,
    596326.4500254393,
    1.0,
    1.0,
    0.972549019607843,
    14795665.69902289,
    0.988235,
    0.980392,
    0.870588,
    35677038.567251325,
    0.992156862745098,
    0.972549019607843,
    0.803921568627451,
    51546874.348982334,
    0.992157,
    0.964706,
    0.713725,
    78275008.82592702,
    0.988235,
    0.956863,
    0.643137,
    120872979.08459747,
    0.980392,
    0.917647,
    0.509804,
    156788963.21235335,
    0.968627,
    0.87451,
    0.407843,
    193540171.8623129,
    0.94902,
    0.823529,
    0.321569,
    220268306.3392564,
    0.929412,
    0.776471,
    0.278431,
    259525283.53246915,
    0.909804,
    0.717647,
    0.235294,
    294605948.1613816,
    0.890196,
    0.658824,
    0.196078,
    323422245.31323636,
    0.878431,
    0.619608,
    0.168627,
    364036504.4443337,
    0.870588,
    0.54902,
    0.156863,
    404650763.575431,
    0.85098,
    0.47451,
    0.145098,
    445265022.7065283,
    0.831373,
    0.411765,
    0.133333,
    485879281.83762586,
    0.811765,
    0.345098,
    0.113725,
    526493540.9687232,
    0.788235,
    0.266667,
    0.0941176,
    567107800.0998205,
    0.741176,
    0.184314,
    0.0745098,
    607722059.2309178,
    0.690196,
    0.12549,
    0.0627451,
    648336318.3620149,
    0.619608,
    0.0627451,
    0.0431373,
    686340409.4987272,
    0.54902,
    0.027451,
    0.0705882,
    719750596.8870386,
    0.470588,
    0.0156863,
    0.0901961,
    757337055.2873727,
    0.4,
    0.00392157,
    0.101961,
    810793354.8864042,
    0.188235294117647,
    0.0,
    0.0705882352941176,
]
particle_electrons_uxLUT.ColorSpace = "Lab"
particle_electrons_uxLUT.NanOpacity = 0.0
particle_electrons_uxLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'particle_electrons_ux'
particle_electrons_uxPWF = GetOpacityTransferFunction("particle_electrons_ux")
particle_electrons_uxPWF.Points = [
    -813777010.3574873,
    0.0,
    0.5,
    0.0,
    810793354.8864042,
    1.0,
    0.5,
    0.0,
]
particle_electrons_uxPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display.Representation = "Surface"
clip1Display.ColorArrayName = ["CELLS", "particle_electrons_ux"]
clip1Display.LookupTable = particle_electrons_uxLUT
clip1Display.SelectTCoordArray = "None"
clip1Display.SelectNormalArray = "None"
clip1Display.SelectTangentArray = "None"
clip1Display.OSPRayScaleFunction = "Piecewise Function"
clip1Display.Assembly = "Hierarchy"
clip1Display.SelectOrientationVectors = "None"
clip1Display.ScaleFactor = 2.0093934867816264e-06
clip1Display.SelectScaleArray = "None"
clip1Display.GlyphType = "Arrow"
clip1Display.GlyphTableIndexArray = "None"
clip1Display.GaussianRadius = 1.0046967433908131e-07
clip1Display.SetScaleArray = [None, ""]
clip1Display.ScaleTransferFunction = "Piecewise Function"
clip1Display.OpacityArray = [None, ""]
clip1Display.OpacityTransferFunction = "Piecewise Function"
clip1Display.DataAxesGrid = "Grid Axes Representation"
clip1Display.PolarAxes = "Polar Axes Representation"
clip1Display.ScalarOpacityFunction = particle_electrons_uxPWF
clip1Display.ScalarOpacityUnitDistance = 6.795294978934044e-07
clip1Display.OpacityArrayName = ["CELLS", "particle_electrons_cpu"]
clip1Display.SelectInputVectors = [None, ""]
clip1Display.WriteLog = ""

# setup the color legend parameters for each legend in this view

# get color legend/bar for particle_electrons_uxLUT in view renderView1
particle_electrons_uxLUTColorBar = GetScalarBar(particle_electrons_uxLUT, renderView1)
particle_electrons_uxLUTColorBar.WindowLocation = "Any Location"
particle_electrons_uxLUTColorBar.Position = [0.7817772778402697, 0.5853658536585366]
particle_electrons_uxLUTColorBar.Title = "particle_electrons_ux"
particle_electrons_uxLUTColorBar.ComponentTitle = ""
particle_electrons_uxLUTColorBar.ScalarBarLength = 0.32999999999999985

# set color bar visibility
particle_electrons_uxLUTColorBar.Visibility = 1

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
animationScene1.AnimationTime = 1.3329221244118056e-13
animationScene1.EndTime = 1.3329221244118056e-13
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
SetActiveSource(pNG1)

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
