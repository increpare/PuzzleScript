

var inputVals = {0 : "U",1: "L",2:"D",3:"R",4:"A",tick:"T",undo:" UNDO ",restart:" RESTART "};

function testFunction(td) {

}

for (var i=0;i<testdata.length;i++) {
	test(
		testdata[i][0], 
		function(num){
			return function(){
				var td = testdata[num];
				var testcode = td[1][0];
				var testinput=td[1][1];
				var testresult=td[1][2];
				var targetlevel=td[1][3];
				var randomseed=td[1][4];
				var audiooutput=td[1][5];
				var input="";
				for (var j=0;j<testinput.length;j++) {
					if (j%5==0 && j>0) {
						input+=" ";
					}
					input += inputVals[testinput[j]];
				}
				var errormessage =  testcode+"\n\n\ninput : "+input;
				errormessage += "\ntargetlevel : "+targetlevel;
				if (randomseed!==undefined) {
					errormessage += "\nrandomseed : "+randomseed;
				}
				if (audiooutput!==undefined) {
					errormessage += "\naudioinput : "+audiooutput.join(";");
				}
				ok(runTest(td[1]),errormessage);
			};
		}(i)
	);
}




for (var i=0;i<errormessage_testdata.length;i++) {
	test(
		"🐛"+errormessage_testdata[i][0], 
		function(num){
			return function(){
				var td = errormessage_testdata[num];
				var testcode = td[1][0];
				var testerrors=td[1][1];
				if (td[1].length!==3){
					throw "Error/Warning message testdata has wrong number of fields, invalid. Accidentally pasted in level recording data?"+"\n\n\n"+testcode;
				}
				var errormessage =  testcode+"\n\n\ndesired errors : "+testerrors;
				ok(runCompilationTest(td[1]),errormessage);
			};
		}(i)
	);
}