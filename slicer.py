import vedo
from vedo import *
from vedo.pyplot import CornerHistogram, histogram

settings.immediate_rendering = False  # faster for multi-renderers

class Slicer3DPlotter(Plotter):
    """
    Generate a rendering window with slicing planes for the input Volume.
    """

    def __init__(
        self,
        volume,
        cmaps=("gist_ncar_r", "hot_r", "bone", "bone_r", "jet", "Spectral_r"),
        clamp=True,
        use_slider3d=False,
        show_histo=True,
        show_icon=True,
        icon_size=0.15,
        icon=None,
        draggable=False,
        at=0,
        **kwargs,
    ):
        """
        Generate a rendering window with slicing planes for the input Volume.

        Arguments:
            cmaps : (list)
                list of color maps names to cycle when clicking button
            clamp : (bool)
                clamp scalar range to reduce the effect of tails in color mapping
            use_slider3d : (bool)
                show sliders attached along the axes
            show_histo : (bool)
                show histogram on bottom left
            show_icon : (bool)
                show a small 3D rendering icon of the volume
            draggable : (bool)
                make the 3D icon draggable
            at : (int)
                subwindow number to plot to

        Examples:
            - [slicer1.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/slicer1.py)

            <img src="https://vedo.embl.es/images/volumetric/slicer1.jpg" width="500">
        """
        ################################
        super().__init__(**kwargs)
        self.at(at)
        ################################

        cx, cy, cz, ch = "dr", "dg", "db", (0.3, 0.3, 0.3)
        if np.sum(self.renderer.GetBackground()) < 1.5:
            cx, cy, cz = "lr", "lg", "lb"
            ch = (0.8, 0.8, 0.8)

        if len(self.renderers) > 1:
            # 2d sliders do not work with multiple renderers
            use_slider3d = True

        self.volume = volume
        box = volume.box().alpha(0.2)
        self.add(box)
        
        volume_axes_inset = vedo.addons.Axes(icon)

        if show_icon:
            if icon:
                icon=icon
            else:
                icon=volume

            self.add_inset(
                icon, volume_axes_inset,
                pos=(0.9, 0.95), size=icon_size, c="w", draggable=draggable,
            )

        # inits
        la, ld = 0.7, 0.3  # ambient, diffuse
        dims = volume.dimensions()
        data = volume.pointdata[0]
        rmin, rmax = volume.scalar_range()
        if clamp:
            hdata, edg = np.histogram(data, bins=50)
            logdata = np.log(hdata + 1)
            # mean  of the logscale plot
            meanlog = np.sum(np.multiply(edg[:-1], logdata)) / np.sum(logdata)
            rmax = min(rmax, meanlog + (meanlog - rmin) * 0.9)
            rmin = max(rmin, meanlog - (rmax - meanlog) * 0.9)

        self.cmap_slicer = cmaps

        self.current_i = None
        self.current_j = None
        self.current_k = int(dims[2] / 2)

        self.xslice = None
        self.yslice = None
        self.zslice = None

        self.zslice = volume.zslice(self.current_k).lighting("", la, ld, 0)
        self.zslice.name = "ZSlice"
        self.zslice.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)
        self.add(self.zslice)

        if show_histo:
            self.histogram = histogram(
                data,
                # title=volume.filename,
                bins=20, logscale=True,
                c=self.cmap_slicer, bg=ch, alpha=1,
                axes=dict(text_scale=2),
            ).clone2d(pos=[-0.8,-0.92], scale=0.4)
            self.add(self.histogram)
        
        #################
        def slider_function_x(widget, event):
            i = int(self.xslider.value)
            if i == self.current_i:
                return
            self.current_i = i
            self.xslice = volume.xslice(i).lighting("", la, ld, 0)
            self.xslice.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)
            self.xslice.name = "XSlice"
            self.remove("XSlice")  # removes the old one
            if 0 < i < dims[0]:
                self.add(self.xslice)
            self.render()

        def slider_function_y(widget, event):
            j = int(self.yslider.value)
            if j == self.current_j:
                return
            self.current_j = j
            self.yslice = volume.yslice(j).lighting("", la, ld, 0)
            self.yslice.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)
            self.yslice.name = "YSlice"
            self.remove("YSlice")
            if 0 < j < dims[1]:
                self.add(self.yslice)
            self.render()

        def slider_function_z(widget, event):
            k = int(self.zslider.value)
            if k == self.current_k:
                return
            self.current_k = k
            self.zslice = volume.zslice(k).lighting("", la, ld, 0)
            self.zslice.cmap(self.cmap_slicer, vmin=rmin, vmax=rmax)
            self.zslice.name = "ZSlice"
            self.remove("ZSlice")
            if 0 < k < dims[2]:
                self.add(self.zslice)
            self.render()

        if not use_slider3d:
            self.xslider = self.add_slider(
                slider_function_x,
                0,
                dims[0],
                title="",
                title_size=0.5,
                pos=[(0.8, 0.12), (0.95, 0.12)],
                show_value=False,
                c=cx,
            )
            self.yslider = self.add_slider(
                slider_function_y,
                0,
                dims[1],
                title="",
                title_size=0.5,
                pos=[(0.8, 0.08), (0.95, 0.08)],
                show_value=False,
                c=cy,
            )
            self.zslider = self.add_slider(
                slider_function_z,
                0,
                dims[2],
                title="",
                title_size=0.6,
                value=int(dims[2] / 2),
                pos=[(0.8, 0.04), (0.95, 0.04)],
                show_value=False,
                c=cz,
            )

        else:  # 3d sliders attached to the axes bounds
            bs = box.bounds()
            self.xslider = self.add_slider3d(
                slider_function_x,
                pos1=(bs[0], bs[2], bs[4]),
                pos2=(bs[1], bs[2], bs[4]),
                xmin=0,
                xmax=dims[0],
                t=box.diagonal_size() / mag(box.xbounds()) * 0.6,
                c=cx,
                show_value=False,
            )
            self.yslider = self.add_slider3d(
                slider_function_y,
                pos1=(bs[1], bs[2], bs[4]),
                pos2=(bs[1], bs[3], bs[4]),
                xmin=0,
                xmax=dims[1],
                t=box.diagonal_size() / mag(box.ybounds()) * 0.6,
                c=cy,
                show_value=False,
            )
            self.zslider = self.add_slider3d(
                slider_function_z,
                pos1=(bs[0], bs[2], bs[4]),
                pos2=(bs[0], bs[2], bs[5]),
                xmin=0,
                xmax=dims[2],
                value=int(dims[2] / 2),
                t=box.diagonal_size() / mag(box.zbounds()) * 0.6,
                c=cz,
                show_value=False,
            )

        #################
        


class Slicer3DTwinPlotter(Plotter):
    """
    Create a window with two side-by-side 3D slicers for two Volumes.

    Example:
        ```python
        from vedo import *
        from vedo.applications import Slicer3DTwinPlotter

        vol1 = Volume(dataurl + "embryo.slc")
        vol2 = Volume(dataurl + "embryo.slc")

        plt = Slicer3DTwinPlotter(
            vol1, vol2, 
            shape=(1, 2), 
            sharecam=True,
            bg="white", 
            bg2="lightblue",
        )

        plt.at(0).add(Text2D("Volume 1", pos="top-center"))
        plt.at(1).add(Text2D("Volume 2", pos="top-center"))

        plt.show(viewup='z')
        plt.at(0).reset_camera()
        plt.interactive().close()
        ```
        ![](https://user-images.githubusercontent.com/32848391/268638466-525114bc-7ce8-480b-9c45-af9ea0d93203.png)
    """

    def __init__(self, vol1, vol2, icon, clamp=True, **kwargs):
        super().__init__(**kwargs)

        cmap = "jet"
        cx, cy, cz = "dr", "dg", "db" # slider colors
        ch = (0.8, 0.8, 0.8)
        ambient, diffuse = 0.7, 0.3   # lighting params

        box1 = vol1.box().alpha(0.1)
        box2 = vol2.box().alpha(0.1)

        self.at(0).add(box1)
        
        volume_axes_inset = vedo.addons.Axes(icon)
        self.add_inset(icon, volume_axes_inset, pos=(0.9, 0.92), size=0.25, c="white", draggable=0)

        self.at(1).add(box2)

        self.add_inset(icon, volume_axes_inset, pos=(0.9, 0.92), size=0.25, c="white", draggable=0)

        dims = vol1.dimensions()
        data = vol1.pointdata[0]
        rmin, rmax = vol1.scalar_range()
        if clamp:
            hdata, edg = np.histogram(data, bins=50)
            logdata = np.log(hdata + 1)
            meanlog = np.sum(np.multiply(edg[:-1], logdata)) / np.sum(logdata)
            rmax = min(rmax, meanlog + (meanlog - rmin) * 0.9)
            rmin = max(rmin, meanlog - (rmax - meanlog) * 0.9)

        def slider_function_x(widget, event):
            i = int(self.xslider.value)
            msh1 = vol1.xslice(i).lighting("", ambient, diffuse, 0)
            msh1.cmap(cmap, vmin=rmin, vmax=rmax)
            msh1.name = "XSlice"
            self.at(0).remove("XSlice")  # removes the old one
            msh2 = vol2.xslice(i).lighting("", ambient, diffuse, 0)
            msh2.cmap(cmap, vmin=rmin, vmax=rmax)
            msh2.name = "XSlice"
            self.at(1).remove("XSlice")
            if 0 < i < dims[0]:
                self.at(0).add(msh1)
                self.at(1).add(msh2)

        def slider_function_y(widget, event):
            i = int(self.yslider.value)
            msh1 = vol1.yslice(i).lighting("", ambient, diffuse, 0)
            msh1.cmap(cmap, vmin=rmin, vmax=rmax)
            msh1.name = "YSlice"
            self.at(0).remove("YSlice")
            msh2 = vol2.yslice(i).lighting("", ambient, diffuse, 0)
            msh2.cmap(cmap, vmin=rmin, vmax=rmax)
            msh2.name = "YSlice"
            self.at(1).remove("YSlice")
            if 0 < i < dims[1]:
                self.at(0).add(msh1)
                self.at(1).add(msh2)

        def slider_function_z(widget, event):
            i = int(self.zslider.value)
            msh1 = vol1.zslice(i).lighting("", ambient, diffuse, 0)
            msh1.cmap(cmap, vmin=rmin, vmax=rmax)
            msh1.name = "ZSlice"
            self.at(0).remove("ZSlice")
            msh2 = vol2.zslice(i).lighting("", ambient, diffuse, 0)
            msh2.cmap(cmap, vmin=rmin, vmax=rmax)
            msh2.name = "ZSlice"
            self.at(1).remove("ZSlice")
            if 0 < i < dims[2]:
                self.at(0).add(msh1)
                self.at(1).add(msh2)

        self.at(0)
        bs = box1.bounds()
        self.xslider = self.add_slider3d(
            slider_function_x,
            pos1=(bs[0], bs[2], bs[4]),
            pos2=(bs[1], bs[2], bs[4]),
            xmin=0,
            xmax=dims[0],
            t=box1.diagonal_size() / mag(box1.xbounds()) * 0.6,
            c=cx,
            show_value=False,
        )
        self.yslider = self.add_slider3d(
            slider_function_y,
            pos1=(bs[1], bs[2], bs[4]),
            pos2=(bs[1], bs[3], bs[4]),
            xmin=0,
            xmax=dims[1],
            t=box1.diagonal_size() / mag(box1.ybounds()) * 0.6,
            c=cy,
            show_value=False,
        )
        self.zslider = self.add_slider3d(
            slider_function_z,
            pos1=(bs[0], bs[2], bs[4]),
            pos2=(bs[0], bs[2], bs[5]),
            xmin=0,
            xmax=dims[2],
            value=int(dims[2] / 2),
            t=box1.diagonal_size() / mag(box1.zbounds()) * 0.6,
            c=cz,
            show_value=False,
        )
        
        self.histogram = histogram(
            data,
            # title=volume.filename,
            bins=50, logscale=True,
            c=cmap, bg=ch, alpha=1,
            axes=dict(text_scale=2),
        ).clone2d(pos=[-0.8,-0.94], scale=0.4)
        self.at(0).add(self.histogram)

        data = vol2.pointdata[0]
        rmin, rmax = vol2.scalar_range()
        if clamp:
            hdata, edg = np.histogram(data, bins=50)
            logdata = np.log(hdata + 1)
            meanlog = np.sum(np.multiply(edg[:-1], logdata)) / np.sum(logdata)
            rmax = min(rmax, meanlog + (meanlog - rmin) * 0.9)
            rmin = max(rmin, meanlog - (rmax - meanlog) * 0.9)

        self.histogram = histogram(
            data,
            # title=volume.filename,
            bins=50, logscale=True,
            c=cmap, bg=ch, alpha=1,
            axes=dict(text_scale=2),
        ).clone2d(pos=[-0.8,-0.94], scale=0.4)
        self.at(1).add(self.histogram)

        slider_function_z(0,0) ## init call
