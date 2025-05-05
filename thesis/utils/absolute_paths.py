import os
import pyprojroot

from glob   import glob
from typing import Union

class AbsolutePaths:
    """
    Class for manipulating absolute paths within the package.

    Attributes
    ----------
    excluded_folders
    excluded_files
    file_extensions_excluded
    parent_path : str
        Absolute package path
    option_paths_folders : dict
        Dictionary with absolute paths of subpackages up to the fifth level 
        of depth
    option_paths_files : dict
        Dictionary with absolute paths of subpackages up to the fifth level 
        of depth
    max_level: int
        Maximum level of depth within the package, in which it seeks to establish 
        the absolute paths


    Methods
    -------
    get_absolute_path(self, folder_name, deep=1)
        Gets the absolute path of the searched folder according to the 
        depth level.
    """

    _excluded_folders = ['.vscode', '__pycache__', 'venv', 'env',
                                '.venv', '.env','.git']
    _excluded_files = ['__init__.py', 'cpython-39.pyc']
    _file_endings_excluded = ['cpython-39.pyc']

    def __init__(self, max_level: int = 5) -> None:
        """
        Parameters
        ----------
        max_level : int, optional
            Maximum level of depth within the parent package, by default 5

        """

        self.parent_path = pyprojroot.here().__str__()
        self.max_level = max_level
        self.__build_paths()

    def get_abs_path_file(self, file_name: str, deep: int = 5) -> Union[str, list]:

        """
        Gets the absolute path of the searched folder according to the depth 
        level.

        Parameters
        ----------
        file_name : str
            Searched file name
        deep : int, optional
            Max depth level to search for the file, by default 5

        Returns
        -------
        str | list
            Path or list of paths for the searched object

        Raises
        ------
        ValueError
            If the passed object name does not exist within the passed deep
        """

        if file_name in self._excluded_files:
            raise ValueError('This file is excluded from searching')

        file_lev = []
        paths_lev = []

        for level in range(0, deep+1):
            for option in self.option_paths_files[level]:
                file_lev.append(option.split(os.sep)[-1])
                paths_lev.append(option)
        
        if file_name in file_lev:
            possible_index = [index for (index, item) in enumerate(file_lev) 
                                if item == file_name]
            
            if len(possible_index) == 1:
                return paths_lev[possible_index[0]]
            else:
                return [paths_lev[index] for index in possible_index]
        else:
            raise ValueError(f"File not found. Explored depth: {deep} levels")

    def get_abs_path_folder(self, folder_name: str, deep: int = 5) -> Union[str, list]:

        """
        Gets the absolute path of the searched folder according to the depth 
        level.

        Parameters
        ----------
        folder_name : str
            Searched folder name
        deep : int, optional
            Max depth level to search for the folder, by default 5

        Returns
        -------
        str | list
            Path or list of paths for the searched object

        Raises
        ------
        ValueError
            If the passed object name does not exist within the passed deep
        """

        if folder_name in self._excluded_folders:
            raise ValueError('This folder is excluded from searching')

        folder_lev = []
        paths_lev = []

        for level in range(1, deep+1):
            for option in self.option_paths_folders[level]:
                folder_lev.append(option.split(os.sep)[-2])
                paths_lev.append(option)
        
        if folder_name in folder_lev:
            possible_index = [index for (index, item) in enumerate(folder_lev) 
                                if item == folder_name]
            
            if len(possible_index) == 1:
                return paths_lev[possible_index[0]]
            else:
                return [paths_lev[index] for index in possible_index]
        else:
            raise ValueError(f"The folder is not found in a depth up to {deep} levels")

    def __build_paths(self) -> dict:
        """
        Build a dictionary with paths for depth levels from 1 to max_level.

        Returns
        -------
        dict
            Dictionary with the absolute paths of each level until the 
            max_level
        """

        def paths_at_level(level: int, type: str = 'folders') -> list:
            """
            Gets the absolute paths of the folders at the searched level.

            Parameters
            ----------
            level : int
                 Level deep into parent directory to get paths
            type : {'folders', 'files'}, optional
                Swith the search between paths of folders or files, 
                by default 'folders'

            Returns
            -------
            list
                List of routes at the level sought

            Raises
            ------
            ValueError
                if you supply some type other than 'folder' or 'file'
            """

            if level == 0:
                options = glob(self.parent_path)
            else: 
                options = glob(self.parent_path+os.sep+f'{os.sep}*'*level)

            if type == 'folders':
                options = [opt+os.sep for opt in options if (
                                os.path.isdir(opt) and 
                                opt.split(os.sep)[-1] not in self._excluded_folders
                            )
                        ]

            elif type == 'files':
                options = [opt for opt in options if (
                                (os.path.isfile(opt)) 
                                and
                                (opt.split(os.sep)[-1] not in self._excluded_files)
                                and
                                not(any([opt.endswith(end) for end in self._file_endings_excluded]))
                            )
                        ]
            else:
                raise ValueError('You can only select between folders and files')

            return options

        dict_paths_folders = {level:paths_at_level(level, type='folders') for level in 
                                                        range(0,self.max_level+1)}
        dict_paths_files = {level:paths_at_level(level, type='files') for level in 
                                                        range(0,self.max_level+1)}

        self.option_paths_folders = dict_paths_folders
        self.option_paths_files = dict_paths_files


    # ----------------------------------------------------------------------
    # attrs


    @property
    def excluded_folders(self) -> list[str]:
        """
        Returns a list with the name of the folders excluded for the construction of absolute paths

        Examples
        --------
        >>> (abs_path.excluded_folders(
        ['.vscode', '__pycache__', 'venv', 'env', '.venv', '.env','.git']
        """

        return self._excluded_folders

    @excluded_folders.setter
    def excluded_folders(self, folder_list: list[str]) -> None:
        """
        Sets the list of folders that are ignored by the instance

        Parameters
        ----------
        folder_list : list
            List of folders that will be ignored when building absolute paths

        Raises
        ------
        ValueError
            If a different value is passed to a list of strings
        """
        
        if not isinstance(folder_list, list):
            raise ValueError("You must pass a list of folders inside the package"
                                " that will be excluded")

        self._excluded_folders = folder_list
        self.__build_paths()

    def add_excluded_folders(self, folder_name: str) -> None:
        """
        Add a new folder to the list of excluded folders.

        Parameters
        ----------
        folder_name : str
            Folder to be ignored for building absolute paths
        """

        self._excluded_folders.append(folder_name)
        self.__build_paths()

    @property
    def excluded_files(self) -> list:
        """
        Returns a list of generic filenames ignored during the construction of 
        absolute paths
        """

        return self._excluded_files

    @excluded_files.setter
    def excluded_files(self, file_list: list[str]) -> None:
        """
        Sets the list of generic files that are ignored by the instance

        Parameters
        ----------
        file_list : list
            List of files that will be ignored when building absolute paths

        Raises
        ------
        ValueError
            If a different object is passed to a list of strings
        """
        if not isinstance(file_list, list):
            raise ValueError("You must pass a list of generic files inside the" 
                                " package that will be excluded")

        self._excluded_files = file_list
        self.__build_paths()


    @property
    def file_endings_excluded(self) -> list:
        """
        Returns a list containing endings of file names that are excluded
        """

        return self._file_endings_excluded
        
    @file_endings_excluded.setter
    def file_endings_excluded(self, ending_list: list[str]) -> None:
        """
        Sets a list of filename endings to be excluded by the instance

        Parameters
        ----------
        ending_list : list
            List of file name endings that will be ignored when building absolute 
            paths

        Raises
        ------
        ValueError
            If a different object is passed to a list of strings
        """
        
        if not isinstance(ending_list, list):
            raise ValueError("You must pass a list of file name endings that are" 
                                " inside the package and that will be ignored.")

        self._file_endings_excluded = ending_list
        self.__build_paths()