from paraview import catalyst
from paraview.simple import *  # noqa: F403


# Helper function
def create_data_extractor(data_node, filename="Dataset"):
    """Creates a data extractor that saves `data_node` to a datafile named `filename`.
    The filetype is chosen based on the type of `data_note`.

    Note: no rendering is performed by such an extractor. The data are
    written directly to a file via VTK.
    """
    VTK_TYPES = [
        "vtkImageData",
        "vtkRectilinearGrid",
        "vtkStructuredGrid",
        "vtkPolyData",
        "vtkUnstructuredGrid",
        "vtkUniformGridAMR",
        "vtkMultiBlockDataSet",
        "vtkPartitionedDataSet",
        "vtkPartitionedDataSetCollection",
        "vtkHyperTreeGrid",
    ]
    FILE_ASSOCIATIONS = [
        "VTI",
        "VTR",
        "VTS",
        "VTP",
        "VTU",
        "VTH",
        "VTM",
        "VTPD",
        "VTPC",
        "HTG",
    ]
    clientside_data = data_node.GetClientSideObject().GetOutputDataObject(
        0
    )  # Gets the dataobject from the default output port

    # Loop is required because .IsA() detects valid classes that inherit from the VTK_TYPES
    for i, vtk_type in enumerate(VTK_TYPES):
        if clientside_data.IsA(vtk_type):
            filetype = FILE_ASSOCIATIONS[i]
            extractor = CreateExtractor(
                filetype, data_node, registrationName=f"_{filetype}"
            )
            extractor.Writer.FileName = filename + "_{timestep:}" + f".{filetype}"
            return extractor

    raise RuntimeError(f"Unsupported data type: {clientside_data.GetClassName()}")


# Camera settings
paraview.simple._DisableFirstRenderCameraReset()  # Prevents the camera from being shown

# Options
options = catalyst.Options()

options.CatalystLiveTrigger = "TimeStep"  # "Python", "TimeStep", "TimeValue"
options.EnableCatalystLive = 0  # 0 (disabled), 1 (enabled)
if options.EnableCatalystLive == 1:
    options.CatalystLiveURL = "localhost:22222"  # localhost:22222 is default

options.ExtractsOutputDirectory = "datasets"  # Base for where all files are saved
options.GenerateCinemaSpecification = 0  # 0 (disabled), 1 (enabled), generates additional descriptor files for cinema exports
options.GlobalTrigger = "TimeStep"  # "Python", "TimeStep", "TimeValue"

meshSource = PVTrivialProducer(
    registrationName="mesh"
)  # "mesh" is the node where the mesh data is stored
create_extractor(meshSource, filename="meshdata")
particleSource = PVTrivialProducer(
    registrationName="particles"
)  # "particles" is the node where particle data is stored
create_extractor(particleSource, filename="particledata")


# Called on catalyst initialize (after Cxx side initialize)
def catalyst_initialize():
    return


# Called on catalyst execute (after Cxx side update)
def catalyst_execute(info):
    print(f"Time: {info.time}, Timestep: {info.timestep}, Cycle: {info.cycle}")
    return


# Callback if global trigger is set to "Python"
def is_activated(controller):
    return True


# Called on catalyst finalize (after Cxx side finalize)
def catalyst_finalize():
    return


if __name__ == "__main__":
    paraview.simple.SaveExtractsUsingCatalystOptions(options)
